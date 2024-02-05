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
        

# layers of VGG16net
# ['', 'features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15', 'features.16', 'features.17', 'features.18', 'features.19', 'features.20', 'features.21', 'features.22', 'features.23', 'features.24', 'features.25', 'features.26', 'features.27', 'features.28', 'features.29', 'features.30', 'avgpool', 'classifier', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6']
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )