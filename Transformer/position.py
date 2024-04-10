import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
all kinds of position encoder 
the postion encoder decede the max length of input sequence 
'''

class CosinPosition(nn.Module):
    def __init__(self,d_model,max_length=2000,dropout=0.1) -> None:
        super(CosinPosition,self).__init__()
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)
        position=torch.zeros(max_length,d_model)
        t=torch.arange(0,max_length,dtype=torch.float).view(-1,1)
        # just to simplify the computation 
        div_term=torch.exp(torch.arange(0,max_length,2,dtype=torch.float)*(-math.log(10000.0)/d_model))
        # a=t*div_term
        # print(a[:,:d_model//2].size())
        position[:,0::2]=torch.sin((t*div_term)[:,:d_model//2])
        position[:,1::2]=torch.cos((t*div_term)[:,:d_model//2])
        # for broadcast
        position.view(1,max_length,d_model)
        self.register_buffer("position",position)

    @torch.no_grad()
    def forward(self,X):
        sentence_length=X.size(1)
        X=X*math.sqrt(self.d_model)
        X=X+self.position[:,:sentence_length]
        X=self.dropout(X)
        return X
        

class RnnPosition(nn.Module):
    '''rnn can replace the cos\sin position encoder to encode the position information.'''
    def __init__(self) -> None:
        super(RnnPosition,self).__init__()

    def forward(self,X):
        return X