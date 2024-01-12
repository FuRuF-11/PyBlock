import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q,k,v,mask=None,dropout=None):
    d_k=k.size(3)
    sorce=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k).float())
    if(mask!=None):
        mask=mask.unsqueeze(1)
        sorce=sorce.masked_fill(mask==0,-1e9)
    att_sorce=F.softmax(sorce,dim=-1)
    if(dropout != None):
        att_sorce=dropout(att_sorce)
    exp=torch.matmul(att_sorce,v)
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

        # attention_sorce.size()=(batch_size,self.head,sentence_length,self.d_head)
        attention_sorce=attention(q,k,v,mask,self.dropout)
        # concat_attention_sorce.size()=(batch_size,sentence_length,self.head,self.d_head)
        concat_attention_sorce=attention_sorce.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output=self.output(concat_attention_sorce)
        return output
    

