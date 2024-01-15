import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
all kinds of position encoder 
the postion encoder decede the max length of input sequence 
'''

class cosinPosition(nn.Module):
    def __init__(self) -> None:
        super(cosinPosition,self).__init__()



    def forward(self,X):
        return X
        