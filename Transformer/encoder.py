import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CasualAttention
from feedforward import MLP
from position import CosinPosition

class EncoderBlock(nn.Module):
    def __init__(self,d_model,sentence_length,head=8,dropout=0.1):
        super(EncoderBlock,self).__init__

    def forward(self,X):
        return X    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder,self).__init__()

    def forward(self,X):
        return X
