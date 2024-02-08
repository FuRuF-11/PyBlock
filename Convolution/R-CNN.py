
# all kinds of R-CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class backbone()

# class Mask()

# class RegionProposalNet()

class MaskRCNN(nn.Module):
    def __init__(self,in_channal,out_channal,kernel_size=3) -> None:
        super().__init__()

    def forward(self,X):
        return 