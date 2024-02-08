import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,X):
        return 