import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# fix needed

# h,w: height and width  of input picture 

class BackBoneNet(nn.Module):
    '''
    FCN need a backbone net to fretch the freature 
    this is just a implementation of a simple CNN,you can train this on a large data set  
    you need to return 3 level feature for unsampling
    if you want use a pre-train model, you can write your own backbone net and sand it to FCN
    '''
    def __init__(self,in_channal,out_channal,picture_ize=(512,512),dropout=0.1) -> None:
        '''
        in_channal and out_channal are just the literal mean
        picture_size accept the  
        '''
        super().__init__()
        self.picture_size=picture_ize
        # [h-->1/2*h]
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channal,1024,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            # pooling to half the size of feature map
            # [batch, channal ,h,w]--->[batch, channal ,h/2,w/2]
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # [1/2*h-->1/4*h]
        self.conv2=nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # [1/4*h-->1/8*h]
        self.conv3=nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # [1/8*h-->1/16*h]
        self.conv4=nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # [1/16*h-->1/32*h]
        tmp=128*int(1/32*self.h*self.h)
        self.mlp=nn.Sequential(
            nn.Linear(tmp,tmp+out_channal),
            nn.ReLU(),
            nn.Linear(tmp+out_channal,out_channal),
            nn.Sigmoid()
        )
    def forward(self,X):
        
        s1=self.conv1(X)
        f1=self.conv2(s1)
        f2=self.conv3(f1)
        f3=self.conv4(f2)
        # [batch,channal,h,w](4D)-->[batch,channal*h*w](2D)
        f3=f3.view(f3.size(0),-1)
        output=self.mlp(f3)
        return output,f1,f2,f3


class FCN(nn.Module):
    def __init__(self,in_channal,out_channal,backbone,kernel_size=3,stride=8,dropout=0.1):
        super().__init__()
        self.class_size
        
    def forward(self,X):
        output=None
        return output
        