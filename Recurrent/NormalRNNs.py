import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
import math


'''
Beacuse the length of sentences are not always the same. We had to pad all sentences to the same length.

In order to avoid the padding part of sentences to influnce the final output of RNN.  

We use rnn_utils.pack_padded_sequence() and rnn_utils.pad_packed_sequence() to pack and pad each sentence.

So, except the sentences X, you also need to input the length of each sentences l
'''

class RNN(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size,dropout=0.1,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.layers=num_layer
        self.bid=bid
        
        self.rnn=nn.RNN(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)

    def forward(self,X,l):
        '''
        X ==> (batch, max_length, d_model) sentences
        l ==> (batch, 1) the true length of each sentences 
        '''
        if(self.bid==True):
            h0=torch.zeros(X.size(0),2*self.num_layer,self.hidden_size)
        else:
            h0=torch.zeros(X.size(0),self.num_layer,self.hidden_size)
        X=rnn_utils.pack_padded_sequence(X,l,batch_first=True,enforce_sorted=False)
        X,hn=self.rnn(X,h0)
        # total_length: the max data length of this batch
        X,_=rnn_utils.pad_packed_sequence(X,batch_first=True)
        return X

class GRU(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size=50,dropout=0.3,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.layer=num_layer
        self.bid=bid
        self.rnn=nn.GRU(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)
        self.tokengenerate=nn.Sequential(
            nn.Linear(in_features=hidden_size,out_features=output_size,bias=True),
        )

    def forward(self,X,l):
        '''
        X ==> (batch, max_length, d_model) sentences
        l ==> (batch, 1) the true length of each sentences 
        '''
        if(self.bid==True):
            h0=torch.zeros(2*self.layer,X.size(0),self.hidden_size)
        else:
            h0=torch.zeros(self.layer,X.size(0),self.hidden_size)
        X=rnn_utils.pack_padded_sequence(X,l,batch_first=True,enforce_sorted=False)
        X,hn=self.rnn(X,h0)
        X,_=rnn_utils.pad_packed_sequence(X,batch_first=True)
        X=torch.stack([X[i,sl-1,:] for i,sl in enumerate(l)],dim=0)
        return X
    
    @torch.inference_mode()
    def inference(self,X):
        '''
        input ==> (1, sentence_length, d_model)
        only one sentence 
        '''
        X=torch.tensor(X).float()
        if(self.bid==True):
            h0=torch.zeros(2*self.layer,self.hidden_size)
        else:
            h0=torch.zeros(self.layer,self.hidden_size)    
        X,_=self.rnn(X,h0)
        return X

class LSTM(nn.Module):
    def __init__(self,d_model,hidden_size,num_layer,output_size=50,dropout=0.3,bid=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.hidden_size=hidden_size
        self.layer=num_layer
        self.bid=bid
        self.rnn=nn.LSTM(d_model,hidden_size,num_layer,\
                        batch_first=True,dropout=dropout,bidirectional=bid)
    
    def forward(self,X,l):
        '''
        X ==> (batch, max_length, d_model) sentences
        l ==> (batch, 1) the true length of each sentences 
        '''
        if(self.bid==True):
            h0=torch.zeros(2*self.layer,X.size(0),self.hidden_size)
            c0=torch.zeros(2*self.layer,X.size(0),self.hidden_size)
        else:
            h0=torch.zeros(self.layer,X.size(0),self.hidden_size)
            c0=torch.zeros(self.layer,X.size(0),self.hidden_size)
        X=rnn_utils.pack_padded_sequence(X,l,batch_first=True,enforce_sorted=False)
        X,(hn,cn)=self.rnn(X,(h0,c0))
        X,_=rnn_utils.pad_packed_sequence(X,batch_first=True)
        X=torch.stack([X[i,sl-1,:] for i,sl in enumerate(l)],dim=0)
        return X
    
    @torch.inference_mode()
    def inference(self,X):
        '''
        input ==> (1, sentence_length, d_model)
        only one sentence 
        '''
        X=torch.tensor(X).float()
        if(self.bid==True):
            h0=torch.zeros(2*self.layer,self.hidden_size)
            c0=torch.zeros(2*self.layer,self.hidden_size)
        else:
            h0=torch.zeros(self.layer,self.hidden_size)
            c0=torch.zeros(self.layer,self.hidden_size)
        X,(hn,cn)=self.rnn(X,(h0,c0))
        return X