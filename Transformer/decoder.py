import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from attention import SelfAttention
from feedforward import MLP
from position import CosinPosition,RnnPosition

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super(DecoderBlock,self).__init__()

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()

class GPT(nn.Module):
    def __init__(self) -> None:
        super(GPT,self).__init__()