import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from bfn.bfn_utils import right_pad_dims_to


class VanillaBFNDiscrete(nn.Module):
    """
    Bayesian Flow Network for discrete data.
    
    Parameters:
    D (int): dimensionality of data
    K (int): number of classes
    network: network used for predictions for p_output
    beta1 (float): initial beta parameter
    """
    def __init__(self, D=2, K=2, model=None, beta1=3.0):
        super(VanillaBFNDiscrete, self).__init__()
        self.D = D
        self.K = K
        self.beta1 = beta1

        output_classes=K if K>2 else 1
        self.model = model


    def forward(self, theta, t):
            """
            
            """
            theta = (theta * 2) - 1 # rescale to [-1, 1] to have distribution centered around 0
            # theta = theta.view(theta.shape[0], -1) # (B, D*K)
            # input_ = torch.cat((theta, t.unsqueeze(-1)), dim=-1) # (B, D*K + 1)
            # output = self.model(input_) # (B, D*K)
            # output =  output.view(output.shape[0], self.D, -1) # (B, D, K)
            output = self.model(theta, t) # (B, D, K)
            return output


    def discrete_output_distribution(self, theta:torch.Tensor, t:torch.Tensor)->torch.Tensor:
            """
            Parameters:
            theta (torch.Tensor): Input tensor of shape (B, D, K).
            t (torch.Tensor): Time tensor of shape (B,).
            
            Returns:
            p_out (torch.Tensor): Output probability tensor. 
                                  If K=2, tensor shape is (B, D, 2). 
                                  If K>2, tensor shape is (B, D, K).
            """
            net_output = self.forward(theta, t) # (B, D, K)

            if self.K == 2:
                po_1 = torch.sigmoid(net_output) # (B, D, K)
                po_2 = 1 - po_1
                p_out = torch.cat((po_1, po_2), dim=-1) # (B, D, 2)

            else:
                p_out = torch.softmax(net_output, dim=-1) # (B, D, K)

            return p_out
            

    def training_continuous_loss(self, x:torch.Tensor):
            B, D = x.shape

            # Sample t~U(0, 1)
            t = torch.rand((B,), device=x.device, dtype=torch.float32) # (B,)

            # Calculate beta
            beta = self.beta1 * (t**2) # (B,)

            # Sample y from p_sender N(beta*(K * e_x - 1), beta * K * I)
            e_x = F.one_hot(x, num_classes=self.K) # (B, D, K)
            mean = beta.unsqueeze(-1).unsqueeze(-1) * (self.K * e_x.float() - 1) # (B, D, K);
            std = (beta.unsqueeze(-1).unsqueeze(-1) * self.K).sqrt() # (B, D, K)
            
            y = torch.distributions.Normal(mean, std).sample() # (B, D, K)

            # Update theta
            theta = torch.softmax(y, dim=-1) # (B, D, K)

            # Calculate p_output
            p_output = self.discrete_output_distribution(theta, t) # (B, D, K)

            # Calculate coninuous Loss
            e_hat = p_output
            Loss_infty = self.K * self.beta1 * t.unsqueeze(-1).unsqueeze(-1) * ((e_x - e_hat)**2) # (B, D, K)

            return Loss_infty.mean()
        

    @torch.inference_mode()
    def sample(self, batch_size:int, n_steps:int, device='cpu'):
            self.eval()

            # prior theta
            theta = torch.ones(size=(batch_size, self.D, self.K), device=device) / self.K # (batch_size, D, K)

            # generation loop
            for i in range(1, n_steps):
                # Calculate t
                t = (i - 1)/n_steps # scalar
                t = t * torch.ones((theta.shape[0],), device=theta.device, dtype=theta.dtype) # (batch_size,)
                
                # Calculate k
                k_probs = self.discrete_output_distribution(theta, t)  # (B, D, K)
                k = torch.distributions.Categorical(probs=k_probs).sample() # (B, D) # should we do Multinomial??

                # Calculate alpha
                alpha = self.beta1 * ((2*i - 1) / n_steps**2) # scalar

                # Sample y from N(alpha*(K * e_k - 1), alpha * K * I)
                e_k = F.one_hot(k, num_classes=self.K).float()  # (B, D, K)
                mean = alpha * (self.K * e_k -1) # (B, D, K)
                var = (alpha * self.K)
                std = torch.full_like(mean, fill_value=var, device=device).sqrt() # (B, D, K)
                y = torch.distributions.Normal(mean, std).sample() # (B, D, K)

                # Update theta
                theta_prime = torch.exp(y) * theta # (B, D, K)
                sum_theta_prime = theta_prime.sum(-1, keepdim=True) # (B, D, 1)
                theta = theta_prime / sum_theta_prime # (B, D, K)
            
            k_probs = self.discrete_output_distribution(theta, torch.ones_like(t))  # (B, D, K)
            k_output_sample = torch.distributions.Categorical(probs=k_probs).sample()
            
            return k_output_sample
    


class BFNDiscrete(nn.Module):
    """
    Bayesian Flow Network for discrete data.
    
    Parameters:
    D (int): dimensionality of data
    K (int): number of classes
    network: network used for predictions for p_output
    beta1 (float): initial beta parameter
    """
    def __init__(self, K=2, model=None, beta1=3.0):
        super(BFNDiscrete, self).__init__()
        #self.D = D
        self.K = K
        self.beta1 = beta1
        self.epsilon = 1e-6

        output_classes=K if K>2 else 1
        self.model = model

    def sample_t_uniformly(self, x: Tensor) -> Tensor:
        t = torch.rand(x.size(0), device=x.device).unsqueeze(-1)
        t = (torch.ones_like(x).flatten(start_dim=1) * t).reshape_as(x)
        return t
    
    @torch.no_grad()
    def params_to_net_inputs(self, params: tuple[Tensor]) -> Tensor:
        params = params[0]
        if self.K == 2:
            params = params * 2 - 1  # We scale-shift here for MNIST instead of in the network like for text
            params = params[..., :1]
        return params

    def get_sender_dist(self, x: Tensor, beta: Tensor) -> D.Distribution:
        e_x = F.one_hot(x.long(), num_classes=self.K) # (B, D, K)
        beta = right_pad_dims_to(beta, e_x) # (B, D, K)
        mean = beta * (self.K * e_x.float() - 1) # (B, D, K);
        std = (beta * self.K).sqrt() # (B, D, K)
            
        return torch.distributions.Normal(mean, std)


    def forward(self, theta:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
            """
            
            """
            # theta = (theta * 2) - 1 # rescale to [-1, 1] to have distribution centered around 0
            # theta = theta.view(theta.shape[0], -1) # (B, D*K)
            # input_ = torch.cat((theta, t.unsqueeze(-1)), dim=-1) # (B, D*K + 1)
            # output = self.model(input_) # (B, D*K)
            # output =  output.view(output.shape[0], self.D, -1) # (B, D, K)
            net_inputs = self.params_to_net_inputs((theta, t)) # if K = 2: (B, D, K) -> (B, D, 1) scaled to [-1,1]
            output = self.model(net_inputs, t) # (B, D, K)
            return output


    def discrete_output_distribution(self, theta:torch.Tensor, t:torch.Tensor)->torch.Tensor:
            """
            Parameters:
            theta (torch.Tensor): Input tensor of shape (B, D, K).
            t (torch.Tensor): Time tensor of shape (B,).
            
            Returns:
            p_out (torch.Tensor): Output probability tensor. 
                                  If K=2, tensor shape is (B, D, 2). 
                                  If K>2, tensor shape is (B, D, K).
            """
            t = right_pad_dims_to(t, theta) # (B, D, K)
            net_output = self.forward(theta, t) # (B, D, K)

            if self.K == 2:
                po_1 = torch.sigmoid(net_output) # (B, D, K)
                po_2 = 1 - po_1
                p_out = torch.cat((po_1, po_2), dim=-1) # (B, D, 2)

            else:
                p_out = torch.softmax(net_output, dim=-1) # (B, D, K)

            return p_out
            

    def continuous_time_loss_for_discrete_data(self, x:torch.Tensor):
            B = x.shape[0]
            #print(f'input x: {x.shape}')
            # Sample t~U(0, 1)
            t = self.sample_t_uniformly(x)
            # flatten data, this makes data that may be 2d such as images Tensor[B, H, W] to flat Tensor[B, H*W]
            x = x.flatten(start_dim=1) # (B, D)
            t = t.flatten(start_dim=1) # (B, D)
            #print(f'falttened x: {x.shape}')

            # Calculate beta
            beta = self.beta1 * (t.clamp(max=1 - self.epsilon)**2) # (B, D)

            # Sample y from p_sender N(beta*(K * e_x - 1), beta * K * I)
            y = self.get_sender_dist(x, beta).sample() # (B, D, K)

            # Update theta
            theta = torch.softmax(y, dim=-1) # (B, D, K)

            # Calculate p_output
            p_output = self.discrete_output_distribution(theta, t) # (B, D, K)

            # Calculate coninuous Loss
            e_x = F.one_hot(x, num_classes=self.K) # (B, D, K)
            e_hat = p_output
            t = right_pad_dims_to(t, e_hat) # (B, D, K)

            Loss_infty = self.K * self.beta1 * t * ((e_x - e_hat)**2) # (B, D, K)

            return Loss_infty.mean()
        

    @torch.inference_mode()
    def sample(self, batch_size:int, n_steps:int, device='cpu'):
            self.eval()

            # prior theta
            theta = torch.ones(size=(batch_size, self.D, self.K), device=device) / self.K # (batch_size, D, K)

            # generation loop
            for i in range(1, n_steps):
                # Calculate t
                t = (i - 1)/n_steps # scalar
                t = t * torch.ones((theta.shape[0],), device=theta.device, dtype=theta.dtype) # (batch_size,)
                
                # Calculate k
                k_probs = self.discrete_output_distribution(theta, t)  # (B, D, K)
                k = torch.distributions.Categorical(probs=k_probs).sample() # (B, D) # should we do Multinomial??

                # Calculate alpha
                alpha = self.beta1 * ((2*i - 1) / n_steps**2) # scalar

                # Sample y from N(alpha*(K * e_k - 1), alpha * K * I)
                e_k = F.one_hot(k, num_classes=self.K).float()  # (B, D, K)
                mean = alpha * (self.K * e_k -1) # (B, D, K)
                var = (alpha * self.K)
                std = torch.full_like(mean, fill_value=var, device=device).sqrt() # (B, D, K)
                y = torch.distributions.Normal(mean, std).sample() # (B, D, K)

                # Update theta
                theta_prime = torch.exp(y) * theta # (B, D, K)
                sum_theta_prime = theta_prime.sum(-1, keepdim=True) # (B, D, 1)
                theta = theta_prime / sum_theta_prime # (B, D, K)
            
            k_probs = self.discrete_output_distribution(theta, torch.ones_like(t))  # (B, D, K)
            k_output_sample = torch.distributions.Categorical(probs=k_probs).sample()
            
            return k_output_sample
            




    