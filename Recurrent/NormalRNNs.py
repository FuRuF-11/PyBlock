import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RNN(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size,dropout=0.1,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.rnn=nn.RNN(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)
        self.classfier=nn.Sequential(
            nn.Linear(hidden_size,output_size),
            nn.ReLU(),
            nn.Softmax(output_size)
        )

    def forward(self,X):
        h0=torch.zeros()
        return 
    


class GRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,X):
        return 
    


class LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,X):
        return 
    