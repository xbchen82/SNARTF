from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: Optional[float] = None):
        '''Implemented simple attention'''
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        self.attn_type = 'entmax15'

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e12)
        if self.attn_type == 'softmax':
            attn = F.softmax(scores, dim=-1)
        elif self.attn_type == 'sparsemax':
            attn = sparsemax(scores, dim=-1)
        elif self.attn_type == 'entmax15':
            attn = entmax15(scores, dim=-1)
        elif self.attn_type == 'entmax':
            attn = entmax_bisect(scores, alpha=1.6, dim=-1, n_iter=25)
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, inputs: int, a_dropout: Optional[float] = None, f_dropout: Optional[float] = None):
        '''Implemented simple multi-head attention'''
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads

        self.attention = ScaledDotProductAttention(a_dropout)
        self.linear_q = nn.Linear(inputs, inputs)
        self.linear_k = nn.Linear(inputs, inputs)
        self.linear_v = nn.Linear(inputs, inputs)
        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs = q.size(0)
        q = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q, k, v, mask).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))


class GraphMultiHeadAttentionEinSum(nn.Module):

    def __init__(self, heads: int, inputs: int, a_dropout: Optional[float] = None, f_dropout: Optional[float] = None,
                 alpha: float = 0.1):
        '''Implemented multi-head spatial attention via multiple stacked Graph Attention Layers (GATs)'''
        super(GraphMultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads
        self.alpha = alpha
        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

        self.alpha = alpha
        self.concat = True
        self.attn_type = 'entmax15'
        self.W1 = nn.Linear(in_features=self.inputs, out_features=self.inputs, bias=False)
        self.W2 = nn.Linear(in_features=self.inputs, out_features=self.inputs, bias=False)
        self.A_i = nn.Linear(in_features=self.hidden, out_features=self.heads, bias=False)
        self.A_j = nn.Linear(in_features=self.hidden, out_features=self.heads, bias=False)
        self.indices = [i for i in range(self.heads)]
        self.dropout = nn.Dropout(p=a_dropout) if a_dropout is not None else nn.Identity()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, num_agent: int, adj: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        Wh1 = torch.stack(torch.chunk(self.W1(h), self.heads, dim=-1), dim=-2)  # [Batch,Temp*Num_agent,Heads,Feature]
        Wh2 = torch.stack(torch.chunk(self.W2(h), self.heads, dim=-1), dim=-2)  # [B,T*N,H,F]
        sWh1 = self.A_i(Wh1)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]
        sWh2 = self.A_j(Wh1)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]
        oWh1 = self.A_i(Wh2)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]
        oWh2 = self.A_j(Wh2)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]

        se = sWh1 + sWh2.transpose(-1, -2)  # [B,H,N*T,N*T]
        oe = oWh1 + oWh2.transpose(-1, -2)  # [B,H,N*T,N*T]
        frame_lengths = h.shape[1] // num_agent
        self_mask = torch.eye(num_agent).repeat(frame_lengths, frame_lengths).to(h)
        self_mask = self_mask.unsqueeze(0).unsqueeze(0)
        e = se * self_mask + oe * (1 - self_mask)
        if not (bias is None):
            biases = torch.stack(bias.chunk(self.heads, -1), dim=1)  # H x [B, H,N, N]
            e = e + biases
        if adj is None:
            adj = torch.ones_like(e).to(e.device)
        attention = e.masked_fill(adj == 0, -1e12)
        if self.attn_type == 'softmax':
            attention = self.dropout(F.softmax(attention, dim=-1))
        elif self.attn_type == 'sparsemax':
            attention = self.dropout(sparsemax(attention, dim=-1))
        elif self.attn_type == 'entmax15':
            attention = self.dropout(entmax15(attention, dim=-1))
        elif self.attn_type == 'entmax':
            attention = self.dropout(entmax_bisect(attention, alpha=1.6, dim=-1, n_iter=25))
        h_prime = torch.matmul(attention * self_mask, Wh1.transpose(1, 2)) + torch.matmul(attention * (1 - self_mask),
                                                                                          Wh2.transpose(1, 2))
        if self.concat:
            h_prime = torch.cat(torch.chunk(h_prime, self.heads, dim=1), dim=-1).squeeze(1)
        return self.dropout(F.elu(self.output(h_prime)))


class GraphMultiHeadAttention(nn.Module):

    def __init__(self, heads: int, inputs: int, a_dropout: Optional[float] = None, f_dropout: Optional[float] = None,
                 alpha: float = 0.1):
        '''Implemented multi-head spatial attention via multiple stacked Graph Attention Layers (GATs)'''
        super(GraphMultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads
        self.alpha = alpha
        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

        self.alpha = alpha
        self.concat = True
        self.attn_type = 'entmax15'
        self.W1 = nn.Linear(in_features=self.inputs, out_features=self.inputs, bias=False)
        self.W2 = nn.Linear(in_features=self.inputs, out_features=self.inputs, bias=False)
        self.A_i = nn.Linear(in_features=self.hidden, out_features=self.heads, bias=False)
        self.A_j = nn.Linear(in_features=self.hidden, out_features=self.heads, bias=False)
        self.indices = [i for i in range(self.heads)]
        self.dropout = nn.Dropout(p=a_dropout) if a_dropout is not None else nn.Identity()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, num_agent: int, adj: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        Wh1 = torch.stack(torch.chunk(self.W1(h), self.heads, dim=-1), dim=-2)  # [Batch,Temp*Num_agent,Heads,Feature]
        Wh2 = torch.stack(torch.chunk(self.W2(h), self.heads, dim=-1), dim=-2)  # [B,T*N,H,F]
        sWh1 = self.A_i(Wh1)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]
        sWh2 = self.A_j(Wh1)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]
        oWh1 = self.A_i(Wh2)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]
        oWh2 = self.A_j(Wh2)[:, :, self.indices, self.indices].transpose(1, 2).unsqueeze(-1)  # [B,H,T*N,1]

        se = sWh1 + sWh2.transpose(-1, -2)  # [B,H,N*T,N*T]
        oe = oWh1 + oWh2.transpose(-1, -2)  # [B,H,N*T,N*T]
        frame_lengths = h.shape[1] // num_agent
        self_mask = torch.eye(num_agent).repeat(frame_lengths, frame_lengths).to(h)
        self_mask = self_mask.unsqueeze(0).unsqueeze(0)
        e = se * self_mask + oe * (1 - self_mask)
        if not (bias is None):
            biases = torch.stack(bias.chunk(self.heads, -1), dim=1)  # H x [B, H,N, N]
            e = e + biases
        if adj is None:
            adj = torch.ones_like(e).to(e.device)
        attention = e.masked_fill(adj == 0, -1e12)
        if self.attn_type == 'softmax':
            attention = self.dropout(F.softmax(attention, dim=-1))
        elif self.attn_type == 'sparsemax':
            attention = self.dropout(sparsemax(attention, dim=-1))
        elif self.attn_type == 'entmax15':
            attention = self.dropout(entmax15(attention, dim=-1))
        elif self.attn_type == 'entmax':
            attention = self.dropout(entmax_bisect(attention, alpha=1.6, dim=-1, n_iter=25))
        h_prime = torch.matmul(attention * self_mask, Wh1.transpose(1, 2)) + torch.matmul(attention * (1 - self_mask),
                                                                                          Wh2.transpose(1, 2))
        if self.concat:
            h_prime = torch.cat(torch.chunk(h_prime, self.heads, dim=1), dim=-1).squeeze(1)
        return self.dropout(F.elu(self.output(h_prime)))
