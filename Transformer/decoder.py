import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from attention import SelfAttention,CasualSelfAttention,MultiHeadAttention
from feedforward import MLP
from position import CosinPosition,RnnPosition

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

def sourceMask(src,pad=0):
        '''
        the length of source sentences are not always same, this is not good for computation.
        to make sequences have the same length, we need to align sentences from left.
        so a real data batch may look like this, in which 'N' means None
        [C,C.C,C,C,N,N]
        [V,V.V,V,N,N,N]
        [K,K,K,K,K,N,N]
        we need to mask N to prevent attention mechanisms from noticing them.
        and that is what this function for
        src: the source sequence
        pad: the pad, could be 0/None/<pad>/...
        '''
        # unsqueeze(-2) to align with the multi-head attention to boardcast
        # src: [batch,sentence]-->2*.unqueeze(1)-->src_mask: [batch,1,1,sentence]
        # att_weight: [batch,head,sentence,sentence]
        src_mask=(src!=pad).unqueeze(1).unqueeze(1)
        return src_mask

class DecoderBlock(nn.Module):
    def __init__(self,d_model,head=8,max_length=2000,dropout=0.1) -> None:
        super(DecoderBlock,self).__init__()
        self.position=CosinPosition(d_model,max_length,dropout)
        self.attention1=CasualSelfAttention(d_model,head,dropout)
        self.attention2=MultiHeadAttention(d_model,head,dropout)
        self.feedforward=MLP(d_model,dropout)
        self.Norm1=nn.LayerNorm(d_model)
        self.Norm2=nn.LayerNorm(d_model)

    def forward(self,X,en_output,src_mask=None):
        
        return X

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()

    def forward(self,X):
        return X

class GPT(nn.Module):
    def __init__(self) -> None:
        super(GPT,self).__init__()