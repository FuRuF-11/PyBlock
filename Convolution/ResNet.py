import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size)

    def forward(self,X,a=1.0):
        return 
class ResNet(nn.Module):
    def __init__(self,in_channel,hidden_channel,output_size,layers):
        super().__init__()
        self.in_channel=in_channel
        self.hidden_channel=hidden_channel
        self.output_size=output_size
        self.layers=layers
        self.conv0=nn.Conv2d(in_channels=in_channel,)
    

    def forward(self,X):
        return