from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadAttention
from .feedforward import FeedForwardNet
import torch.nn.functional as F


class DecoderLayer(nn.Module):

    def __init__(self, inputs: int, heads: int, hidden: int, a_dropout: Optional[float] = None,
                 f_dropout: Optional[float] = None):
        '''Implemented decoder layer via multi-head attention and feedforward net'''
        super(DecoderLayer, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden

        self.self_attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)

        self.self_attention_norm = nn.LayerNorm(inputs)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, \
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        y = self.attention_norm(tgt)
        y = self.attention(y, src, src, mask=src_mask)
        tgt = tgt + y

        y = self.feedforward_norm(tgt)
        y = self.feedforward(y)
        tgt = tgt + y

        return tgt


class Decoder(nn.Module):

    def __init__(self, layers: int, inputs: int, heads: int, hidden: int, a_dropout: Optional[float] = None,
                 f_dropout: Optional[float] = None):
        '''Implemented decoder via multiple stacked decoder layers'''
        super(Decoder, self).__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.attention_dropout = a_dropout
        self.feature_dropout = f_dropout

        self.norm = nn.LayerNorm(inputs)
        layers = [DecoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout) for _ in range(layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, \
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        return self.norm(tgt)


def init_weight(modules):
    # initialize embeddings and norm layers
    for module in modules:
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module)


