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
    def __init__(self,d_model,hidden_size,head=8,dropout=0.1):
        super(EncoderBlock,self).__init__()
        self.Norm1=nn.LayerNorm(d_model)
        self.Norm2=nn.LayerNorm(d_model)
        self.attention=SelfAttention(d_model,head,dropout)
        self.feedforward=MLP(d_model,hidden_size,dropout)

    def forward(self,X,mask=None):
        X=X+self.attention(self.Norm1(X),mask)
        print(X.size())
        print("------------")
        # batch, sentence_length, d_model
        X=X+self.feedforward(self.Norm2(X))
        print(X.size())
        return X
    
class Encoder(nn.Module):
    def __init__(self,config) -> None:
        super(Encoder,self).__init__()
        self.layer_size=config["layer_size"]
        self.position=CosinPosition(config["d_model"],config["sentence_length"],config["dropout"])
        self.layers=cloneLayers(EncoderBlock(config["d_model"],config["hidden_size"],config["head_num"],config["dropout"]),self.layer_size)
        self.Norm=nn.LayerNorm(config["d_model"])

    def forward(self,X,mask=None):
        X=self.position(X)
        for i in range(self.layer_size):
            X=self.layers[i](X,mask)
        X=self.Norm(X)
        return X

class Bert(nn.Module):
    def __init__(self) -> None:
        super(Bert,self).__init__()