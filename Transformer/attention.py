import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q,k,v,mask=False):
    sorce=torch.matmul()
    att_sorce=None
    exp=None
    return exp 

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head,dropout=0.1) -> None:
        super(MultiHeadAttention,self).__init__()
        self.d_head=d_model//head
        self.d_model=d_model
        self.K=nn.Linear()
        self.Q=nn.Linear()
        self.V=nn.Linear()
        self.output=nn.Linear()
        self.dropout=dropout

    def forward(self,q,k,v,mask=None):
        '''
        mask is a matrix you need to buiild by yourself
        if you want  to use self attention ,just let q=k=v=YourInput
        '''
        output=self.output()
        return output
    

