import torch
import torch.nn as nn
import torch.nn.functional as F

'''
we only implemet the multi-head version attention ,because the single-head is just a speical version of multi-head
'''

def attention(q,k,v,mask=None,dropout=None):
    '''
    compute attention score 
    '''
    d_k=k.size(3)
    # sorce.size()=(batch_size,self.head,sentence_length,sentence_length)
    score=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k).float())
    if(mask!=None):
        score=score.masked_fill(mask==0,float('-inf'))
    att_score=F.softmax(score,dim=-1)
    if(dropout != None):
        att_score=dropout(att_score)
    exp=torch.matmul(att_score,v)
    return exp 

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head,dropout=0.1) -> None:
        super(MultiHeadAttention,self).__init__()
        self.head=head
        self.d_head=d_model//head
        self.d_model=d_model
        self.K=nn.Linear(d_model,d_model)
        self.Q=nn.Linear(d_model,d_model)
        self.V=nn.Linear(d_model,d_model)
        self.output=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        '''
        mask is a matrix you need to buiild by yourself or just use casual attention
        if you want  to use self attention ,just let q=k=v=YourInput
        '''
        # q.size()=(batch_size,sentence_length,self.d_model)
        batch_size=q.size(0)
        # q.size()=(batch_size,self.head,sentence_length,self.d_head)
        q=self.q(q).view(batch_size,-1,self.head,self.d_head).transpose(1,2)
        k=self.k(k).view(batch_size,-1,self.head,self.d_head).transpose(1,2)
        v=self.v(v).view(batch_size,-1,self.head,self.d_head).transpose(1,2)

        # attention_score.size()=(batch_size,self.head,sentence_length,self.d_head)
        attention_score=attention(q,k,v,mask,self.dropout)
        # concat_attention_sorce.size()=(batch_size,sentence_length,self.d_model)
        # view() will not change the order of numbers in memory,so we can use it to concat different heads.
        # [[a,b],[c,d]] --->[[a,b,c,d]]
        concat_attention_score=attention_score.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output=self.output(concat_attention_score)
        return output

class SelfAttention(nn.Module):
    '''
    self attention version multi-head attention
    '''
    def __init__(self,d_model,head,dropout=0.1) -> None:
        super(SelfAttention,self).__init__()
        self.MultiHeadAtt=MultiHeadAttention(d_model=d_model,head=head,dropout=dropout)

    def forward(self,X,mask=None):
        output=self.MultiHeadAtt(X,X,X,mask)
        return output


class CasualSelfAttention(nn.Module):
    ''' 
    an implmentation of masked multi-head self attention 
    which is often used in NLP models like GPT.
    the length and width of mask matrix equal to the length of sentence 
    '''
    def __init__(self,d_model,head,sentence_length,dropout=0.1) -> None:
        super(CasualSelfAttention).__init__()
        self.register_buffer(\
            'mask',torch.tril(torch.ones(sentence_length,sentence_length)).\
            view(1,1,sentence_length,sentence_length))
        self.SelfAttention=SelfAttention(d_model=d_model,head=head,dropout=dropout)

    def forward(self,X):
        output=self.SelfAttention(X,self.mask)
        return output
