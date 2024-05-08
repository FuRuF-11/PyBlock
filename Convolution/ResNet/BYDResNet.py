import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def cloneLayers(layer,n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class Block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride) -> None:
        super().__init__()
        self.block=nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.BatchNorm2d()
                )
    def forward(self,X):
        return self.block(X)

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,block_num,stride=1) -> None:
        super().__init__()
        self.block_num=block_num
        self.block1=Block(in_channel,out_channel,kernel_size,stride=stride)
        self.blocks=cloneLayers(Block(out_channel,out_channel,kernel_size),self.block_num-1)
            
    def forward(self,X):
        X=X+self.block1(X)
        for i in range(self.block_num-1):
            X=X+self.blocks[i](X)
        return X

class ResNet18(nn.Module):
    def __init__(self,output_size,in_channel):
        super().__init__()
        self.output_size=output_size
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv2=ConvBlock(64,64,3,2)
        self.conv3=ConvBlock(64,128,3,2,stride=2)
        self.conv4=ConvBlock(128,256,3,2,stride=2)
        self.conv5=ConvBlock(256,512,3,2,stride=2)
        self.avepool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear()
        )
    def forward(self,X):
        return