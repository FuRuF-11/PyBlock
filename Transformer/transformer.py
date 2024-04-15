import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder


@torch.no_grad()
def sourceMask(src1,src2=None):
        '''
        the length of source sentences are not always same, this is not good for computation.
        to make sequences have the same length, we need to align sentences from left.
        so a real data batch may look like this, in which 'N' means None
        [C, C, C, N, N, N, N]
        [V, V. V, V, N, N, N]
        [K, K, K, K, K, K, N]
        we need to mask N to prevent attention mechanisms from noticing them.
        and that is what this function for
        src: the source sequence
        pad: the pad, could be 0/None/<pad>/...
        '''
        # unsqueeze(-2) to align with the multi-head attention to boardcast
        # src: [batch,sentence]-->2*.unqueeze(1)-->src_mask: [batch,1,1,sentence]
        # att_weight: [batch,head,sentence,sentence]
        src_mask=(src1.sum(dim=2)!=0).float()
        
        # .unsqueeze(-2).unsqueeze(1)

        # a=src_mask
        
        # print(a.size())
        # print(a)
        # print(src_mask)
        # sc1.size(1),sc2.size()
        # src_mask=[]
        return src_mask


class Transformer(nn.Module):
    def __init__(self,config) -> None:
        '''
        input config should contian at least these args
        d_model: the dims of every vector
        hidden_size: the hidden size of hidden layer
        sentence_length: the length of sentence
        layer_size=6: the layer size of encoder/decoder
        head=8: the head size of multi-head attention
        pad=0: the pad of src_mask,could be 0/None/<pad>
        dropout=0.1: the dropout rate
        '''
        super(Transformer,self).__init__()
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
        # self.output=nn.Linear(config["hidden_size"],config["output_size"],bias=True)

    def forward(self,source_seq,traget_seq):
        src_mask=sourceMask(source_seq)
        en_output=self.encoder(source_seq,src_mask)
        de_output=self.decoder(traget_seq,en_output,src_mask)
        # output=self.output(de_output)
        return de_output

