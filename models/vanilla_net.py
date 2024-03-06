import torch
from torch import nn

class VanillaNetDiscretized(nn.Module):
    """
    Vanilla neural network for discretized data.
    
    Parameters:
    D (int): dimensionality of data
    K (int): number of classes
    hidden_dim (int): number of hidden units
    output_classes (int): number of output classes
    """
    def __init__(self, D=1, hidden_dim=128):
        super(VanillaNetDiscretized, self).__init__()
        self.D = D
        self.hidden_dim = hidden_dim

        self.layer = nn.Sequential(
                nn.Linear(D + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, D * 2)
                )
        
    def forward(self, theta, t):
        input = torch.cat([theta, t], dim=-1) # (B, D+1)
        output = self.layers(input)  
        output = output.view(output.shape[0], self.D, 2) # (B, D, 2)
        return output


class VanillaNetDiscrete(nn.Module):
    """
    Vanilla neural network for discrete data.
    """
    def __init__(self, D=1, K=2, hidden_dim=128):
        super(VanillaNetDiscrete, self).__init__()
        self.D = D
        self.K = K
        self.hidden_dim = hidden_dim

        output_classes=K if K>2 else 1

        self.layer = nn.Sequential(
                nn.Linear(D * K + 1 , hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, D * output_classes)
                )   
        
    def forward(self, theta, t):
        theta = theta.view(theta.shape[0], -1) # (B, D*K)
        input_ = torch.cat((theta, t.unsqueeze(-1)), dim=-1) # (B, D*K + 1)
        output = self.layer(input_) # (B, D*K)
        output =  output.view(output.shape[0], self.D, -1) # (B, D, K)

        return output