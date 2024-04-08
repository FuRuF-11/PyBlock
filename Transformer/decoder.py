import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .attention import SelfAttention,CasualSelfAttention,MultiHeadAttention
from .feedforward import MLP
from .position import CosinPosition,RnnPosition

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class DecoderBlock(nn.Module):
    def __init__(self,d_model,head=8,dropout=0.1) -> None:
        super(DecoderBlock,self).__init__()
        self.attention1=CasualSelfAttention(d_model,head,dropout)
        self.attention2=MultiHeadAttention(d_model,head,dropout)
        self.feedforward=MLP(d_model,dropout)
        self.Norm1=nn.LayerNorm(d_model)
        self.Norm2=nn.LayerNorm(d_model)
        self.Norm3=nn.LayerNorm(d_model)

    def forward(self,X,en_output,src_mask=None):
        X=X+self.attention1(self.Norm1(X))
        x=self.Norm2(X)
        X=X+self.attention2(X,en_output,en_output,src_mask)
        X=X+self.feedforward(self.Norm3(X))
        return X

class Decoder(nn.Module):
    def __init__(self,d_model,layer_size=6,head=8,dropout=0.1,max_length=2000) -> None:
        super(Decoder,self).__init__()
        self.N=layer_size
        self.position=CosinPosition(d_model,max_length,dropout)
        self.layers=cloneLayers(DecoderBlock(d_model,head,dropout),layer_size)
        self.Norm=nn.LayerNorm(d_model)

    def forward(self,X,en_output,src_mask=None):
        X=X+self.position(X)
        for i in range(self.N):
            X=self.layers[i](X,en_output,src_mask)
        X=self.Norm(X)
        return X

class GPT(nn.Module):
    def __init__(self) -> None:
        super(GPT,self).__init__()