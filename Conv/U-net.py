# U-net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,inChannl,outChannl,kernelSize) -> None:
        super(ConvBlock,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(inChannl,outChannl,kernel_size=kernelSize,padding=1,bias=True),
            nn.BatchNorm2d(outChannl),
            nn.ReLU(),
            nn.Conv2d(outChannl,outChannl,kernel_size=kernelSize,padding=1,bias=True),
            nn.BatchNorm2d(outChannl),
            nn.ReLU()
        )
    def forward(self,X):
        X=self.conv(X)
        return X

class up_conv(nn.Module):
    def __init__(self) -> None:
        super(up_conv,self).__init__()
        # dim 3&4 scale twice
        self.upsample=nn.Upsample(scale_factor=2)
        # 

    def forward(self,X):
        return X

class U_net(nn.Module):
    def __init__(self,layer) -> None:
        super(U_net,self).__init__()

    def forward(self,X):
        
        return X