from typing import Optional
import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GraphMultiHeadAttention
from .feedforward import FeedForwardNet
from .positional import PositionalEncoding


class EncoderLayer(nn.Module):

    def __init__(self, inputs: int, heads: int, hidden: int, a_dropout: Optional[float] = None,
                 f_dropout: Optional[float] = None):
        '''Implemented encoder layer via multi-head self-attention and feedforward net'''
        super(EncoderLayer, self).__init__()
        self.heads = heads
        self.hidden = hidden
        self.inputs = inputs

        self.attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.attention_norm(x)
        y = self.attention(y, y, y, mask=mask)
        x = x + y

        y = self.feedforward_norm(x)
        y = self.feedforward(y)
        x = x + y

        return x


class GraphEncoderLayer(nn.Module):

    def __init__(self, inputs: int, heads: int, hidden: int, a_dropout: Optional[float] = None,
                 f_dropout: Optional[float] = None, alpha: float = 0.1):
        '''Implemented spatial encoder layer via multi-head spatial self-attention and feedforward net'''
        super(GraphEncoderLayer, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden

        self.attention = GraphMultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout, alpha=alpha)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, h: torch.Tensor, num_agent: int, adj: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.attention_norm(h)
        y = self.attention(y,num_agent, adj, bias)
        h = h + y

        y = self.feedforward_norm(h)
        y = self.feedforward(y)
        h = h + y
        return h


class Encoder(nn.Module):

    def __init__(self, layers: int, inputs: int, heads: int, hidden: int, a_dropout: Optional[float] = None,
                 f_dropout: Optional[float] = None):
        '''Implemented encoder via multiple stacked encoder layers'''
        super(Encoder, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden
        self.attention_dropout = a_dropout
        self.feature_dropout = f_dropout

        self.norm = nn.LayerNorm(inputs)
        layers = [EncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout) for _ in range(layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x  # self.norm(x)


class GraphEncoder(nn.Module):

    def __init__(self, layers: int, inputs: int, heads: int, hidden: int, a_dropout: Optional[float] = None,
                 f_dropout: Optional[float] = None, alpha: float = 0.1):
        '''Implemented spatial encoder via multiple stacked spatial encoder layers'''
        super(GraphEncoder, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden
        self.attention_dropout = a_dropout
        self.feature_dropout = f_dropout
        self.alpha = alpha

        self.norm = nn.LayerNorm(inputs)
        layers = [GraphEncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout, alpha=alpha) for _
                  in range(layers)]
        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor, num_agent: int, adj: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, num_agent, adj, bias)
        return self.norm(h)
