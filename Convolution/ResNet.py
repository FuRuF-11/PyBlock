import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class ResNet(nn.Module):
    def __init__(self,in_channel,hidden_channel,output_size,layers):
        super().__init__()
        self.in_channel=in_channel
        self.hidden_channel=hidden_channel
        self.output_size=output_size
        self.layers=layers
    

    def forward(self,X):
        return