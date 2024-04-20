import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import sourceMask
from .encoder import Encoder
from .decoder import Decoder   

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
        self.config=config
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
        # self.output=nn.Linear(config["hidden_size"],config["output_size"],bias=True)

    def forward(self,src_seq,trg_seq):
        mask1=sourceMask(src1=src_seq)
        mask2=sourceMask(src1=trg_seq,src2=src_seq)
        en_output=self.encoder(src_seq,mask1)
        de_output=self.decoder(trg_seq,en_output,mask2)
        # output=self.output(de_output)
        return de_output
    
    @torch.no_grad()
    def generate(self,sentnece1,sentnece2):
        '''
        we need two different sentences to run the transformr
        and the second sentnece need to be finished 
        '''
        s2l=sentnece2.size(0)
        sentnece2=F.pad(sentnece2,(0,0,0,self.config["trg_length"]-s2l),value=0)
        sentnece1=sentnece1.view(1,sentnece1.size(0),sentnece1.size(1))
        sentnece2=sentnece2.view(1,sentnece2.size(0),sentnece2.size(1))

        mask1=sourceMask(src1=sentnece1)
        en_output=self.encoder(sentnece1,mask1)
        for i in range(self.config["trg_length"]-s2l):
            mask2=sourceMask(src1=sentnece2,src2=sentnece1)
            de_output=self.decoder(sentnece2,en_output,mask2)
            if(torch.sum(de_output[:,-1]==self.config["end_word"])):
                return sentnece2
            sentnece2[0,s2l+i]=de_output[:,-1]
        return sentnece2
