import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .attention import SelfAttention
from .feedforward import MLP
from .position import CosinPosition

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class EncoderBlock(nn.Module):
    def __init__(self,d_model,head=8,dropout=0.1):
        super(EncoderBlock,self).__init__
        self.Norm1=nn.LayerNorm(d_model)
        self.Norm2=nn.LayerNorm(d_model)
        self.attention=SelfAttention(d_model,head,dropout)
        self.feedforward=MLP(d_model,dropout)

    def forward(self,X,mask=None):
        X=X+self.attention(self.Norm1(X),mask)
        X=X+self.feedforward(self.Norm2(X))
        return X
    
class Encoder(nn.Module):
    def __init__(self,d_model,layer_size=6,head=8,dropout=0.1,max_length=2000) -> None:
        super(Encoder,self).__init__()
        self.N=layer_size
        self.position=CosinPosition(d_model,max_length,dropout)
        self.layers=cloneLayers(EncoderBlock(d_model,head,dropout),self.N)
        self.Norm=nn.LayerNorm(d_model)

    def forward(self,X,mask=None):
        X=self.position(X)
        for i in range(self.N):
            X=self.layers[i](X,mask)
        X=self.Norm(X)
        return X

class Bert(nn.Module):
    def __init__(self) -> None:
        super(Bert,self).__init__()