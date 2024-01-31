import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BackBoneNet(nn.Module):
    '''
    FCN need a backbone net to fretch the freature 
    you need to return 3 level feature for unsampling
    if you want use a pre-train model,you need write your own backbone net and sand it to FCN
    '''
    def __init__(self,in_channal,out_channal) -> None:
        super().__init__()

    def forward(self,X):
        f1,f2,f3=(0,0,0)
        return f1,f2,f3


class FCN(nn.Module):
    def __init__(self,in_channal,out_channal,backbone,stride=8):
        super().__init__()
        self.class_size
        
    def forward(self,X):
        output=None
        return output
        