
# all kinds of R-CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class backbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Mask(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class RegionProposalNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class MaskRCNN(nn.Module):
    def __init__(self,in_channal,out_channal,kernel_size=3,pre_train=False) -> None:
        super().__init__()
        if(pre_train==False):
            pass
        else:
            pass

    def forward(self,X):
        return 