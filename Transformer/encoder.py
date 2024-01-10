import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self,X) -> None:
        super(AttentionBlock,self).__init__()
        self.W=nn.Embedding()

class TransformerEncoderBlock(nn.Module):
    def __init__(self,in_channal,out_channal,head):
        super(TransformerEncoderBlock,self).__init__
