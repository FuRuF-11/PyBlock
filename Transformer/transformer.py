import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder


@torch.no_grad()
def sourceMask(src,pad=0):
        '''
        the length of source sentences are not always same, this is not good for computation.
        to make sequences have the same length, we need to align sentences from left.
        so a real data batch may look like this, in which 'N' means None
        [C,C.C,C,C,N,N]
        [V,V.V,V,N,N,N]
        [K,K,K,K,K,N,N]
        we need to mask N to prevent attention mechanisms from noticing them.
        and that is what this function for
        src: the source sequence
        pad: the pad, could be 0/None/<pad>/...
        '''
        # unsqueeze(-2) to align with the multi-head attention to boardcast
        # src: [batch,sentence]-->2*.unqueeze(1)-->src_mask: [batch,1,1,sentence]
        # att_weight: [batch,head,sentence,sentence]
        src_mask=(src!=pad).unqueeze(1).unqueeze(1)
        return src_mask

class Transformer(nn.Module):
    def __init__(self,d_model,layer_size=6,head=8,pad=0,dropout=0.1,max_length=2000) -> None:
        '''
        d_model: the dims of every vector
        layer_size=6: the layer size of encoder/decoder
        head=8: the head size of multi-head attention
        pad=0: the pad of src_mask,could be 0/None/<pad>
        dropout=0.1: the dropout rate
        max_length=2000: the maximum allowed length of sentences in position encoder
        '''
        super(Transformer,self).__init__()
        self.encoder=Encoder(d_model,layer_size,head,dropout,max_length)
        self.decoder=Decoder(d_model,layer_size,head,dropout,max_length)
        self.output=nn.Linear(d_model,d_model,bias=True)
        self.pad=pad

    def forward(self,source_seq,traget_seq):
        en_output=self.encoder(source_seq)
        src_mask=sourceMask(source_seq,self.pad)
        de_output=self.decoder(traget_seq,en_output,src_mask)
        output=self.output(de_output)
        return output
    

class Transformer(nn.Module):
    def __init__(self,config) -> None:
        '''
        d_model: the dims of every vector
        layer_size=6: the layer size of encoder/decoder
        head=8: the head size of multi-head attention
        pad=0: the pad of src_mask,could be 0/None/<pad>
        dropout=0.1: the dropout rate
        max_length=2000: the maximum allowed length of sentences in position encoder
        '''
        super(Transformer,self).__init__()
        self.encoder=Encoder(self,config)
        self.decoder=Decoder(self,config)
        self.output=nn.Linear(config["hidden_size"],config["output_size"],bias=True)
        self.pad=config["pad"]

    def forward(self,source_seq,traget_seq):
        en_output=self.encoder(source_seq)
        src_mask=sourceMask(source_seq,self.pad)
        de_output=self.decoder(traget_seq,en_output,src_mask)
        output=self.output(de_output)
        return output

