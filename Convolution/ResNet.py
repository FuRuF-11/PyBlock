import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 网络参数设置
'''
18
{
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [3x3 64, 3x3 64]x2,
    conv3: [3x3 128, 3x3 128]x2,
    conv4: [3x3 256, 3x3 256]x2,
    conv5: [3x3 512, 3x3 512]x2,
    average pool: 1000-d fc, softmax
}'''

'''
34
{
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [3x3 64, 3x3 64]x3,
    conv3: [3x3 128, 3x3 128]x4,
    conv4: [3x3 256, 3x3 256]x6,
    conv5: [3x3 512, 3x3 512]x3,
    average pool: 1000-d fc, softmax
}'''

'''
50
{
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x4,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x6,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}'''

'''
101
{
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x4,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x23,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}'''

'''
152
{
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x8,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x36,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}'''


class ResNet(nn.Module):
    def __init__(self,in_channel,hidden_channel,output_size,layers):
        super().__init__()
        self.in_channel=in_channel
        self.hidden_channel=hidden_channel
        self.output_size=output_size
        self.layers=layers
        self.conv0=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        

    def forward(self,X):
        return