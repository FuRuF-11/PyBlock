# U-net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class U_net(nn.Module):
    def __init__(self) -> None:
        super(U_net,self).__init__()

    def forward(self,X):
        
        return X