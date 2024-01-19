import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from attention import SelfAttention
from feedforward import MLP
from position import CosinPosition


def cloneLayers(layer,n):
    return nn.ModuleList(copy.deepcopy(layer))


class EncoderBlock(nn.Module):
    def __init__(self,d_model,head=8,dropout=0.1):
        super(EncoderBlock,self).__init__
        self.Norm1=nn.LayerNorm(d_model)
        self.Norm2=nn.LayerNorm(d_model)
        self.attention=SelfAttention(d_model,head,dropout)
        self.feedforward=MLP(d_model,dropout)

    def forward(self,X):
        X=X+self.attention(self.Norm1(X))
        X=X+self.feedforward(self.Norm2(X))
        return X
    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder,self).__init__()

    def forward(self,X):
        return X
