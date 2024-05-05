
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
50 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x4,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x6,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}
'''
'''
101 {
    conv1: 7x7 64 stride 2,
           3x3 maxpool stride 2,
    conv2: [1x1 64, 3x3 64, 1x1 256]x3,
    conv3: [1x1 128, 3x3 128, 1x1 512]x4,
    conv4: [1x1 256, 3x3 256, 1x1 1024]x23,
    conv5: [1x1 512, 3x3 512, 1x1 2048]x3,
    average pool: 1000-d fc, softmax
}
'''
'''
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