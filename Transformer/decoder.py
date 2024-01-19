import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from attention import SelfAttention
from feedforward import MLP
from position import CosinPosition

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super(DecoderBlock,self).__init__()
