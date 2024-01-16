import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
all kinds of position encoder 
the postion encoder decede the max length of input sequence 
'''

class cosinPosition(nn.Module):
    def __init__(self,d_model,max_length=2000,dropout=0.1) -> None:
        super(cosinPosition,self).__init__()
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)
        position=torch.zeros(1,1,max_length,d_model)

    @torch.no_grad()
    def forward(self,X):
        return X
        

class RnnPosition(nn.Module):
    '''rnn can replace the cos\sin position encoder to encode the position information.'''
    def __init__(self) -> None:
        super(RnnPosition,self).__init__()

    def forward(self,X):
        return X