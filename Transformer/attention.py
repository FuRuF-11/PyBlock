import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q,k,v,mask=False):
    sorce=torch.matmul()
    att_sorce=None
    exp=None
    return exp 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,head,dropout=0.1) -> None:
        super(MultiHeadSelfAttention,self).__init__()
        self.d_head=d_model//head
        self.d_model=d_model
        self.K=nn.Linear()
        self.Q=nn.Linear()
        self.V=nn.Linear()

        self.dropout=nn.Dropout(dropout)

    def forward(self,X):
        return X
    

