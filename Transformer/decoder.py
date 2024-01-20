import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from attention import SelfAttention,CasualSelfAttention,MultiHeadAttention
from feedforward import MLP
from position import CosinPosition,RnnPosition

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class DecoderBlock(nn.Module):
    def __init__(self,d_model,head=8,max_length=2000,dropout=0.1) -> None:
        super(DecoderBlock,self).__init__()
        self.position=CosinPosition(d_model,max_length,dropout)
        self.attention1=CasualSelfAttention(d_model,head,dropout)
        self.attention2=MultiHeadAttention(d_model,head,dropout)
        self.feedforward=MLP(d_model,dropout)
        self.Norm1=nn.LayerNorm(d_model)
        self.Norm2=nn.LayerNorm(d_model)

    def forward(self,X,en_output):
        return X

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()

class GPT(nn.Module):
    def __init__(self) -> None:
        super(GPT,self).__init__()