import torch
import torch.nn as nn
from torch.autograd import Variable

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''Implemented positional encoding layer'''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if not (dropout is None) else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, begin: int = 0) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, x.size(-2), x.size(-1))
        x = x + Variable(self.pe[:, begin:begin+x.size(1)], 
                         requires_grad=False)
        x = self.dropout(x)
        x = x.view(*shape)
        return x
