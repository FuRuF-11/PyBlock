import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# warning!!!!!!!!
# this is just a example.
# this is a implementation of FCN with a backbone net from VGG16, which has five maxpoolings.
# if you want to use your own backbone net, you need to pay attention to those things.
# First, the ConfirstvTranspose2d() operation is related with the pooling operation in FCN
# everytime you want use a ConvTranspose2d(), you need to apply a pooling operation to decrease the feature size a half first
# for we want to construct the segmented picture from feature map which only has a half size of what we want
# Second, you can choose different features to reconstruct the segmented picture from pooling
# in this example, we use all five features from five maxpooling operation   

class VGG16net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=torchvision.models.vgg16(pretrained=True)
        



    def forward(self,X):
        return 

class FCN(nn.Module):
    def __init__(self,in_channel,out_channel,backbone=None,kernel_size=3,stride=8,dropout=0.1):
        '''
        the backbone need to  return at least four feature f0,f1,f2,f3
        and the default set is ResNet18 offering by torchvision which has been pretrained 
        '''
        super().__init__()
        self.backbone=VGG16net()
        self.in_channal=in_channel
        self.out_channal=out_channel
        self.Upsample=[nn.Sequential(
            nn.ConvTranspose2d(64*i,128*i,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64*i)
            ) for i in range(1,5)]
        
        self.Segment=nn.ConvTranspose2d(1024,out_channel,kernel_size=1)

    def forward(self,X):
        tmp=[out for idx,out in enumerate(self.backbone(X))]
        for i,elem in enumerate(tmp):
            s=self.Upsample[i](elem)
            if(i!=len(tmp)):
                s=s+tmp[i+1]
        output=self.Segment(s)
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