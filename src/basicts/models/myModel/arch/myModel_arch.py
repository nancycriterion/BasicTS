import torch
from ..config.myModel_config import MyModelConfig
from torch import nn
from .myModel_layers import MyUKF

class myModel(nn.Module):
    def __init__(self,config:MyModelConfig):
        super(myModel,self).__init__()
        self.ukf=MyUKF(config)
        
    
    def forward(self,input:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, output_len, num_features]
        """
        batch_size, input_len, num_features = input.shape