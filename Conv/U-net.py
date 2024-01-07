# U-net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,inChannal,outChannal,kernelSize=3) -> None:
        super(ConvBlock,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(inChannal,outChannal,kernel_size=kernelSize,padding=1,bias=True),
            nn.BatchNorm2d(outChannal),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannal,outChannal,kernel_size=kernelSize,padding=1,bias=True),
            nn.BatchNorm2d(outChannal),
            nn.ReLU(inplace=True)
        )
    def forward(self,X):
        X=self.conv(X)
        return X

class up_conv(nn.Module):
    def __init__(self,inChannl,outChannal,kernelSize=3) -> None:
        super(up_conv,self).__init__()
        # dim 3&4 scale twice
        self.up_and_conv=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inChannl,outChannal,kernel_size=kernelSize,padding=1,bias=True),
            nn.BatchNorm2d(outChannal),
            nn.ReLU(inplace=True)
        )
    def forward(self,X):
        X=self.up_and_conv(X)
        return X

class U_net(nn.Module):
    def __init__(self,inChannal,outChannal,C_size=64,kernel_size=3) -> None:
        super(U_net,self).__init__()
        a=64
        self.maxPooling=nn.MaxPool2d()
        self.conv1=ConvBlock(inChannal,a,kernel_size)
        self.conv2=ConvBlock(a,a*2,kernel_size)
        self.conv3=ConvBlock(a*2,a*4,kernel_size)
        self.conv4=ConvBlock(a*4,a*8,kernel_size)
        self.conv5=ConvBlock(a*8,a*16,kernel_size)
        self.conv6=ConvBlock(a*16,a*8,kernel_size)
        self.conv7=ConvBlock(a*8,a*4,kernel_size)
        self.conv8=ConvBlock(a*4,a*2,kernel_size)
        self.conv9=ConvBlock(a*2,a,kernel_size)
        # every out Channal is a class
        self.conv10=nn.Conv2d(a,outChannal,kernel_size=1)

        self.up1=up_conv(a*16,a*8,kernel_size)
        self.up2=up_conv(a*8,a*4,kernel_size)
        self.up3=up_conv(a*4,a*2,kernel_size)
        self.up4=up_conv(a*2,a,kernel_size)

    def forward(self,X):
        # encoder
        
        return X