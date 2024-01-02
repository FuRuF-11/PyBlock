import torch
import torch.nn as nn
import torch.nn.functional as function


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,dilation,A=False):
        super(CausalConv1d, self).__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.A=A
        # 最优padding方式，可以根据是否是A层，来决定是否对当前的数据进行卷积
        self.padding=(kernel_size-1)*dilation+A*1
        self.conv1d=nn.Conv1d(in_channels,out_channels,kernel_size,\
                              stride=1,padding=0,dilation=dilation)
        # self.padding = (kernel_size - 1, 0)  # 在序列的左侧进行padding，右侧不进行padding
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding,0))  # 序列左侧padding
        conv_out=self.conv1d(x)
        if(self.A):
            return conv_out[:,:,:-1]
        else:
            return conv_out
