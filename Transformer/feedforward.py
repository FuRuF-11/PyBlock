import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''the implementation of all kinds of feedforward blocks '''

class NewGELU(nn.Module):
    '''
    The activation funcation of GPT and Bert
    Reference: 
    Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    On the GELU Activation Function: https://alaaalatif.github.io/2019-04-11-gelu/
    '''
    def __init__(self) -> None:
        super(NewGELU,self).__init__()
    def forward(self,X):
        return 0.5*X*(1.0+torch.tanh(math.sqrt(2/math.pi)*(X+0.044715*X**3)))


class MLP(nn.Module):
    def __init__(self,in_feature,hidden_size,dropout=0.1) -> None:
        super(MLP,self).__init__()
        self.MLP=nn.Sequential(
            nn.LayerNorm(in_feature),
            nn.Linear(in_features=in_feature,out_features=hidden_size),
            NewGELU(),
            nn.Linear(in_features=hidden_size,out_features=in_feature),
            nn.Dropout(dropout)
        )
    def forward(self,X):
        X=self.MLP(X)
        return X
    
class switchGLU(nn.Module):
    def __init__(self,in_feature,hidden_size,dropout=0.1) -> None:
        super().__init__()
        
    def forward(self,X):
        return 