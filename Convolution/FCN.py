import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=torchvision.models.resnet18(pretrained=True)
        self.perpare= nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )
        self.layer1=self.model.layer1
        self.layer2=self.model.layer2
        self.layer3=self.model.layer3
        self.layer4=self.model.layer4

    @torch.no_grad()
    def forward(self,X):
        X=self.perpare(X)
        f0=self.layer1(X)
        f1=self.layer2(f0)
        f2=self.layer3(f1)
        f3=self.layer4(f2)
        return f0,f1,f2,f3


class FCN(nn.Module):
    def __init__(self,in_channal,out_channal,backbone=None,kernel_size=3,stride=8,dropout=0.1):
        '''
        the backbone need to  return at least four feature f0,f1,f2,f3
        and the default set is ResNet18 offering by torchvision which has been pretrained 
        '''
        super().__init__()
        if backbone==None:
            self.backbone=ResNet18()
        else:
            self.backbone=backbone
        self.in_channal=in_channal
        self.out_channal=out_channal
        # in pytorch, the unsample is defult 
        self.Upsample0=nn.Sequential(
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.Upsample1=nn.Sequential(
            nn.ConvTranspose2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.Upsample2=nn.Sequential(
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.Upsample3=nn.Sequential(
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

    def forward(self,X):
        tmp=[None for i in range(4)]
        tmp[0],tmp[1],tmp[2],tmp[3]=self.backbone(X)
        for i in range():
            pass
        output=None
        return output
        

# set layer name set of resnet18
# conv1
# bn1
# relu
# maxpool
# layer1
# layer1.0
# layer1.0.conv1
# layer1.0.bn1
# layer1.0.relu
# layer1.0.conv2
# layer1.0.bn2
# layer1.1
# layer1.1.conv1
# layer1.1.bn1
# layer1.1.relu
# layer1.1.conv2
# layer1.1.bn2
# layer2
# layer2.0
# layer2.0.conv1
# layer2.0.bn1
# layer2.0.relu
# layer2.0.conv2
# layer2.0.bn2
# layer2.0.downsample
# layer2.0.downsample.0
# layer2.0.downsample.1
# layer2.1
# layer2.1.conv1
# layer2.1.bn1
# layer2.1.relu
# layer2.1.conv2
# layer2.1.bn2
# layer3
# layer3.0
# layer3.0.conv1
# layer3.0.bn1
# layer3.0.relu
# layer3.0.conv2
# layer3.0.bn2
# layer3.0.downsample
# layer3.0.downsample.0
# layer3.0.downsample.1
# layer3.1
# layer3.1.conv1
# layer3.1.bn1
# layer3.1.relu
# layer3.1.conv2
# layer3.1.bn2
# layer4
# layer4.0
# layer4.0.conv1
# layer4.0.bn1
# layer4.0.relu
# layer4.0.conv2
# layer4.0.bn2
# layer4.0.downsample
# layer4.0.downsample.0
# layer4.0.downsample.1
# layer4.1
# layer4.1.conv1
# layer4.1.bn1
# layer4.1.relu
# layer4.1.conv2
# layer4.1.bn2
# avgpool
# fc