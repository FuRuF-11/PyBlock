import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNN(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size,dropout=0.1,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.layers=num_layer
        self.bid=bid
        
        self.rnn=nn.RNN(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)
        self.classfier=nn.Sequential(
            nn.Linear(hidden_size,output_size),
            nn.ReLU(),
            nn.Softmax(output_size)
        )

    def forward(self,X):
        if(self.bid==True):
            h0=torch.zeros(X.size(0),2*self.num_layer,self.hidden_size)
        else:
            h0=torch.zeros(X.size(0),self.num_layer,self.hidden_size)
        X,h_n=self.rnn(X,h0)
        out=self.classfier(X[:,-1,:])
        return out
    
class GRU(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size,dropout=0.1,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.layers=num_layer
        self.bid=bid
        
        self.rnn=nn.GRU(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)
        self.classfier=nn.Sequential(
            nn.Linear(hidden_size,output_size),
            nn.ReLU(),
            nn.Softmax(output_size)
        )

    def forward(self,X):
        if(self.bid==True):
            h0=torch.zeros(X.size(0),2*self.num_layer,self.hidden_size)
        else:
            h0=torch.zeros(X.size(0),self.num_layer,self.hidden_size)
        X,h_n=self.rnn(X,h0)
        out=self.classfier(X[:,-1,:])
        return out


class LSTM(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size,dropout=0.1,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.layers=num_layer
        self.bid=bid
        
        self.rnn=nn.LSTM(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)
        self.classfier=nn.Sequential(
            nn.Linear(hidden_size,output_size),
            nn.ReLU(),
            nn.Softmax(output_size)
        )

    def forward(self,X):
        if(self.bid==True):
            h0=torch.zeros(X.size(0),2*self.num_layer,self.hidden_size)
        else:
            h0=torch.zeros(X.size(0),self.num_layer,self.hidden_size)
        X,h_n=self.rnn(X,h0)
        out=self.classfier(X[:,-1,:])
        return out
