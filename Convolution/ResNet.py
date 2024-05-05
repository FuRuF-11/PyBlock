import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 网络参数设置
'''
18 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [3x3 64, 3x3 64]x2,
    conv3: [3x3 128, 3x3 128]x2,
    conv4: [3x3 256, 3x3 256]x2,
    conv5: [3x3 512, 3x3 512]x2,
    average pool: 1000-d fc, softmax
}
34 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [3x3 64, 3x3 64]x3,
    conv3: [3x3 128, 3x3 128]x4,
    conv4: [3x3 256, 3x3 256]x6,
    conv5: [3x3 512, 3x3 512]x3,
    average pool: 1000-d fc, softmax
}
50 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x4,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x6,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}
101 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x4,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x23,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}
152 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x8,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x36,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}
'''


class Block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,layer_num=2) -> None:
        super().__init__()
        if(layer_num==2):
            pass
        elif(layer_num==3):
            pass
    def forward(self):
        return 

class ResNet18(nn.Module):
    def __init__(self,output_size,in_channel):
        super().__init__()
        self.output_size=output_size
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv2_1=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d()
        )
        self.conv2_2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d()
        )
        self.conv3_1=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d()
        )
        self.conv3_2=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d()
        )

    def forward(self,X):
        return