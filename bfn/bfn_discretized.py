import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import functools

from bfn.bfn_const import CONST_log_range, CONST_log_min, CONST_summary_rescale, CONST_exp_range, CONST_min_std_dev
from bfn.bfn_utils import safe_exp, safe_log, right_pad_dims_to
#from torch.utils.tensorboard import SummaryWriter


class BayesianFlowNetworkDiscretized(nn.Module):
    """
    Bayesian Flow Network for discretized data.
    
    Parameters:
    D (int): dimensionality of data
    K (int): number of classes
    network: network used for predictions for p_output
    beta1 (float): initial beta parameter
    """
    def __init__(self, D=1, K=16, model=None, sigma_1=1e-3, device='cpu'):
        super(BayesianFlowNetworkDiscretized, self).__init__()
        self.D = D
        self.K = K # num_bins
        self.sigma_1 = sigma_1
        self.bin_width = 2.0 / K
        self.half_bin_width = self.bin_width / 2.0
        self.t_min = 1e-6
        self.min_variance = 1e-6
        self.model = model
        self.device = device
        self.k_centres = torch.arange(self.half_bin_width - 1, 1, self.bin_width) # (K,)
        self.k_lefts = self.k_centres - 1/self.K # (K,)
        self.k_rights = self.k_centres + 1/self.K # (K,)

    
    def get_gamma(self, t:Tensor):
        return 1 - self.sigma_1**(2 * t)


    def forward(self, mu:Tensor, t:Tensor) -> tuple[Tensor]:
        mu = (mu * 2) - 1 # rescale to [-1, 1] to have distribution centered around 0 # (B, D)
                            # (B, D)  #(B, 1) 
        input = torch.cat([mu, t], dim=-1) # (B, D + 1)
        output = self.model(input) # (B, D * 2) mu and log sigma assume the net learns this 
        output =  output.view(output.shape[0], self.D, -1) # (B, D, 2)
        mu_e = output[:, :, 0] # (B, D)
        log_sigma_e = output[:, :, 1] # (B, D)
        return mu_e, log_sigma_e
        

    def discretised_cdf(self, mu, sigma, x):
        """
        Compute the CDF of the discretised output distribution.
        params:
        mu (torch.Tensor): mean of the output distribution. Shape (B)
        sigma: standard deviation of the output distribution Shape (B)
        x: the input value Shape (B,)
        return:
        G: the CDF of the discretised output distribution. Shape (B)
        """
        # CDF
        F_x = 0.5 * (1 + torch.erf((x - mu)/ (sigma* torch.sqrt(torch.tensor(2.0))) )) # (B,)

        # Clipped CDF for d = 1, kc
        G = torch.where(x <= -1, torch.zeros_like(mu), 
                                 torch.where(x>=1, torch.ones_like(mu), F_x) ) # (B,)
        return G


    def discretised_output_distribution(self, mu:Tensor, t:Tensor, gamma:Tensor):

        mu_e, log_sigma_e = self.forward(mu, t)  # (B, D), (B, D)

        mu_x = (mu / gamma) - (torch.sqrt((1 - gamma) / gamma) * mu_e)
        sigma_x = torch.clamp(torch.sqrt((1 - gamma) / gamma) * safe_exp(log_sigma_e), self.min_variance)

        mu_x = torch.where(t < self.t_min, torch.zeros_like(mu_x), mu_x)  # (B, D)
        sigma_x = torch.where(t < self.t_min, torch.ones_like(sigma_x), sigma_x)  # (B, D)

        # p_output = torch.zeros((mu_x.shape[0], self.D, self.K), device=mu_x.device, dtype=mu_x.dtype)  # (B, D, K)
        # for d in range(self.D):
        #     for i in range(self.K):
        #         kl = self.k_lefts[i]
        #         kr = self.k_rights[i]
                
        #         p = self.discretised_cdf(mu_x[:, d], sigma_x[:, d], kr) - self.discretised_cdf(mu_x[:, d], sigma_x[:, d], kl)
        #         p_output[:, d, i] = p
        
        # # Calculate the discretised output distribution u.to(self.device)sing vectorized operations
        # print("mu", mu_x[:10], "sigma", sigma_x[:10])
        normal_dist = torch.distributions.Normal(mu_x, sigma_x)
        cdf_values_lower = normal_dist.cdf(self.k_lefts)
        # ensure first bin has area 0
        cdf_values_lower = torch.where(self.k_lefts<=-1, torch.zeros_like(cdf_values_lower), cdf_values_lower)
        cdf_values_upper = normal_dist.cdf(self.k_rights)
        # ensure last bin has area 1
        cdf_values_upper = torch.where(self.k_rights>=1, torch.ones_like(cdf_values_upper), cdf_values_upper)

        p_output = cdf_values_upper - cdf_values_lower
                 
        return p_output # (B, D, K)


    def training_continuous_loss(self, x:torch.Tensor):
        #print(f'x.shape = {x.shape}')
        B = x.shape[0]

        # Sample t~U(0, 1)
        t = torch.rand((B, 1), device=x.device, dtype=torch.float32) # (B, 1)
        gamma = self.get_gamma(t) # (B,1)

        # sample from sender distribution Ps(x*gamma, gamma(1-gamma)* I)
        sender_mu_sample = torch.distributions.Normal(x*gamma, gamma*(1-gamma)).sample() # (B, D) 

        # calculate the output distribution
        p_output = self.discretised_output_distribution(sender_mu_sample, t, gamma) # (B, D, K)

        # calculate the receiver distribution k_hat
        k_hat = torch.sum(p_output*self.k_centres, dim=-1) # (B, D)

        # calculate the KL divergence for continuous loss
                                # scalar             # scalar  # (B, 1)      # (B, D)
        Loss_infinity = - np.log(self.sigma_1) * (self.sigma_1**(-2*t)) * ((x-k_hat)**2) # (B, D)

        loss = torch.mean(Loss_infinity)
        return loss
    
    @torch.inference_mode()
    def sample(self, batch_size:int, n_steps:int):
        mu = torch.zeros((batch_size, self.D), dtype=torch.float32) # (batch_size, D)
        ro = 1. #scalar
        for i in range(1, n_steps+1):
            t = (i - 1)/n_steps
            t = t * torch.ones((mu.shape[0], 1), dtype=mu.dtype) # (batch_size, 1)
            gamma = self.get_gamma(t) # (batch_size, 1)
            p_output = self.discretised_output_distribution(mu, t, gamma) # (B, D, K)

            alpha = (self.sigma_1**(-2.0 * i/n_steps)) * (1 - self.sigma_1**(2 / n_steps)) # scalar

            k_hat = torch.sum(p_output*self.k_centres, dim=-1) # (B, D)

            # sample from receiver distribution
                                        # (B, D)  # scalar
            y = torch.distributions.Normal(k_hat, torch.sqrt(alpha**(-1))).sample() # (B, D)

            # update mu
            mu = ((mu * ro) + (alpha * y))/(ro + alpha) # (B, D)
            # update ro
            ro = ro + alpha   # scalar

        t_ones = torch.ones((mu.shape[0],1), dtype=mu.dtype) # (batch_size, 1)
        p_output = self.discretised_output_distribution(mu, t_ones, gamma=torch.tensor(1-self.sigma_1**2)) # (B, D, K)
        k_hat = torch.sum(p_output*self.k_centres, dim=-1) # (B, D)
        return k_hat
    
    @torch.inference_mode()
    def sample_generation_for_discretised_data(self, bs=64, n_steps=20):

        # initialise prior with uniform distribution
        prior_mu = torch.zeros(bs, self.D)
        prior_precision = torch.tensor(1)

        prior_tracker = torch.zeros(bs, self.D, 2, n_steps+1)
        # iterate over n_steps
        for i in range(1, n_steps+1):

            # SHAPE B,1 time is set to fraction from
            t = (i - 1)/n_steps
            t = t * torch.ones((prior_mu.shape[0], 1), dtype=prior_mu.dtype) # (batch_size, 1)
            # SHAPE B,1
            # gamma = self.get_gamma_t(t)

            # B x D x K
            output_distribution = self.discretised_output_distribution(prior_mu, t, gamma=(1-(self.sigma_1 ** (2*t))))

            # SHAPE scalar
            alpha = (self.sigma_1**(-2.0 * i/n_steps)) * (1 - self.sigma_1**(2 / n_steps)) # scalar

            # sample from y distribution centered around 'k centers'

            # B, 1
            eps = torch.randn(bs).to(self.device)
            # SHAPE B x D
            mean = torch.sum(output_distribution*self.k_centres, dim=-1)
            y_sample = mean + (np.sqrt((1/alpha)) * eps)

            # update our prior precisions and means w.r.t to our new sample
            prior_tracker[:, :, 0, i] = y_sample.unsqueeze(1)
            prior_tracker[:, :, 1, i] = prior_precision

            # SHAPE B x D
            # prior_mu = (prior_precision*prior_mu.squeeze() + alpha*y_sample) / (prior_precision + alpha)
            # prior_mu = prior_mu.unsqueeze(1)
            # # shape scalar
            # prior_precision = alpha + prior_precision
            prior_mu, prior_precision = self.update_input_params((prior_mu.squeeze(), prior_precision), y_sample, alpha)
            prior_mu = prior_mu.unsqueeze(1)

        # final pass of our distribution through the model to get final predictive distribution
        output_distribution = self.discretised_output_distribution(prior_mu, torch.ones_like(t), gamma=torch.Tensor([1-self.sigma_1**2]).unsqueeze(-1))
        print(output_distribution.shape)
        # SHAPE B x D
        output_mean = torch.sum(output_distribution*self.k_centres, dim=-1)

        return output_mean, prior_tracker
    
    def update_input_params(self, input_params, y, alpha):
        input_mean, input_precision = input_params
        new_precision = input_precision + alpha
        new_mean = ((input_precision * input_mean) + (alpha * y)) / new_precision
        # print(y.shape, new_mean.shape, new_precision.shape, input_mean.shape, input_precision.shape)
        return new_mean, new_precision


