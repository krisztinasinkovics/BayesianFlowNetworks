from datasets.mnist import get_mnist_dataloaders
from datasets.dataset_utils import get_image_grid_from_tensor
from typing import Union
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from models.unet_improved import UNetModel
from torch.optim import AdamW
from bfn.bfn_discrete import BFNDiscrete
from models.adapters import FourierImageInputAdapter, OutputAdapter
import torch
import wandb
import math
import numpy as np

class DiscreteBFNTrainer():

    def __init__(self,
                 K:int = 2,
                 device: str = None,
                 bs: int = 32,
                 num_epochs: int = 10,
                 input_height: int = 28, # 32 for cifar
                 input_channels: int = 1, # 3 for cifar
                #  add_pos_feats: bool = False,
                #  output_height: int = 2,
                 lr: float = 0.0002,
                 betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.01,
                 model: Union[None, nn.Module] = None,
                 optimizer: Union[None, torch.optim.Optimizer] = None,
                 dataset: str = 'mnist',
                 wandb_project_name: str = "bfn",
                 checkpoint_file: str = None,
                 checkpoint_save_path: str = './bfn_model_checkpoint'):
       
        self.K = K
        self.device = device
        # self.input_height = input_height
        # self.input_channels = input_channels
        self.best_val_loss = torch.tensor(float('inf'))
        self.step = 0
        self.epoch = 0
        self.num_epochs = num_epochs
        self.checkpoint_save_path = checkpoint_save_path

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # load dataset
        if dataset == 'mnist':
            self.train_dts, self.val_dts, self.test_dts = get_mnist_dataloaders()
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
        # init model
        if model is None:
            data_adapters = {
                "input_adapter": FourierImageInputAdapter(input_channels=1, 
                                                          input_shape=[28, 28],
                                                          output_height=2, # # 2 in the mnist_discrete.yaml?
                                                          add_pos_feats=False),
                # model output, rgb channels, 2 model outputs (mean and std)
                "output_adapter": OutputAdapter(network_height=256,
                                                output_channels=1,
                                                n_outputs=1),
            } 
            self.net = UNetModel(data_adapters, 
                                 image_size=28,
                                 in_channels=2, # 2 in the mnist_discrete.yaml?
                                 num_res_blocks=2,
                                 dropout=0.5,
                                 channel_mult=[1, 2, 2],
                                 project_input=True,
                                 )
        else:
            self.net = model

        # init BFN model
        self.bfn_model = BFNDiscrete(K=K, model=self.net).to(self.device)

        # init optimizer
        if optimizer is None:
            self.optim = AdamW(self.bfn_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        steps_per_epoch = len(self.train_dts) 
        total_steps = self.num_epochs * steps_per_epoch
        # self.lr_sched = OneCycleLR(self.optim, max_lr, total_steps=total_steps, pct_start=0.001)

        # load checkpoint if provided
        if checkpoint_file is not None:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file)
            # Load each part
            self.bfn_model.load_state_dict(checkpoint['model'])
            # self.lr_sched = checkpoint['lr_sched']
            self.optim.load_state_dict(checkpoint['optim'])
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            print(f'Loaded pretrained model checkpoint {checkpoint_file}')

        # init ema
        self.ema = ExponentialMovingAverage(self.bfn_model.parameters(), decay=0.9999)

        self.wandb_project_name = wandb_project_name
        if wandb_project_name is not None:
            wandb.init(project=wandb_project_name)


    def train(self,
              num_epochs: int = None,
              validation_interval_epoch: int = 1,
              sampling_interval_step: int = 250,
              save_checkpoint_interval_step: int = 100,
              clip_grad: float = 2.0,
              n_test_batches: int = 0):
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        for i in range(num_epochs):
            epoch_losses = []

            # run through training batches
            for bi, batch_Xy in enumerate(self.train_dts): 
                print(f"Epoch {i+1}/{num_epochs}, Batch {bi+1}/{len(self.train_dts)}")
                batch = batch_Xy[0] # train_dts returns a tuple (inputs, targets), we only use inputs

                if n_test_batches != 0:
                    if bi > n_test_batches:
                        break


                self.optim.zero_grad()

                # model inference
                loss = self.bfn_model.continuous_time_loss_for_discrete_data(batch.to(self.device))

                loss.backward()
                # clip grads
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.bfn_model.parameters(), max_norm=clip_grad)

                # update steps
                self.optim.step()
                self.ema.update()

                # logging
                if self.wandb_project_name is not None:
                    wandb.log({"batch_train_loss": loss.item(), "lr": self.optim.param_groups[0]['lr']})
                print(f"Epoch {i+1}/{num_epochs}, Loss: {torch.mean(torch.tensor(epoch_losses))}")

                epoch_losses.append(loss.item())

                # Sampling stage
                if self.step % sampling_interval_step == 0:
                    self.sample()

                if self.step % save_checkpoint_interval_step == 0:
                    self.save_model(save_path=self.checkpoint_save_path)

                # for every batch iterate
                self.step += 1


            if self.wandb_project_name is not None:
                wandb.log({"epoch_train_loss": torch.mean(torch.tensor(epoch_losses))})
            print(f"Epoch {i+1}/{num_epochs}, Loss: {torch.mean(torch.tensor(epoch_losses))}")

            # Validation check
            if i % validation_interval_epoch == 0:
                self.validate()

        self.epoch += 1 
                
            


    @torch.no_grad()
    def validate(self):
        self.bfn_model.eval()
        val_losses = []

        for _, batch_Xy in enumerate(self.val_dts):
            batch = batch_Xy[0] 

            loss = self.bfn_model.continuous_time_loss_for_discrete_data(batch.to(self.device))
            val_losses.append(loss.item())

        if self.wandb_project_name is not None:
            wandb.log({"validation_loss": torch.mean(torch.tensor(val_losses))})

        epoch_val_loss = torch.mean(torch.tensor(val_losses))
        if self.wandb_project_name is not None:
            wandb.log({"validation_loss": epoch_val_loss})

        if epoch_val_loss < self.best_val_loss:
            self.best_val_loss = epoch_val_loss
            self.save_model(save_path=self.checkpoint_save_path+'best_val')


    @torch.inference_mode()
    def sample(self, sample_shape = (8, 28, 28, 1), n_steps=100):
        self.bfn_model.eval()
        
        # Generate samples and priors
        with self.ema.average_parameters():
            samples = self.bfn_model.sample(sample_shape=sample_shape, n_steps=n_steps)
            samples = samples.to(torch.float32)
        image_grid = get_image_grid_from_tensor(samples.transpose(1, 3)) #samples
        # Convert samples to numpy arrays
        image_grid = image_grid.detach().numpy()
        image_grid = np.transpose(image_grid, (2, 1, 0))
        # priors_np = priors.detach().numpy()
        
        # Plot histograms
        if self.wandb_project_name is not None:
            images = wandb.Image(image_grid, caption="MNIST - Sampled Images from BFN")
            wandb.log({"image_samples": images})

    
    def save_model(self, save_path: str = './bfn_model_checkpoint'):
        self.bfn_model.eval()

        checkpoint = { 
            'epoch': self.epoch,
            'step': self.step,
            'model': self.bfn_model.state_dict(),
            'optim': self.optim.state_dict(),
            }
        # 'lr_sched': self.lr_sched
        save_path = save_path + '.pth'
        print(save_path)
        torch.save(checkpoint, save_path)