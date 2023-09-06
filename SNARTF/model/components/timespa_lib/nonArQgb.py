from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.autograd import Variable


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''Implemented positional encoding layer'''
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if not (dropout is None) else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inputs: Tensor) -> torch.Tensor:
        bsize, _, n, _ = inputs.shape
        x = Variable(self.pe[:, :],
                     requires_grad=False)
        x = x.repeat(bsize, 1, n, 1)
        x = self.dropout(x)
        return x


class TimeEmbeddingSine(nn.Module):
    def __init__(self,
                 d_model: int = 64,
                 temperature: int = 10000,
                 scale: Optional[float] = None,
                 requires_grad: bool = False):
        super(TimeEmbeddingSine, self).__init__()

        self.d_model = d_model
        self.temperature = temperature
        self.scale = 2 * math.pi if scale is None else scale
        self.requires_grad = requires_grad

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.clone()
        d_embed = self.d_model
        dim_t = torch.arange(d_embed, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_embed)
        x = x / dim_t
        x = torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(-2)
        return x if self.requires_grad else x.detach()


class NonAutoRegression(nn.Module):

    def __init__(self, feature_dims: int,  pred_seq_len: int,d_model,
                 dropout: Optional[torch.Tensor] = None):
        super(NonAutoRegression, self).__init__()
        # self.index_pos_embed = TimeEmbeddingSine()
        self.index_pos_embed = PositionalEmbedding(d_model=feature_dims, dropout=dropout, max_len=pred_seq_len)
        self.tgt_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True)
        )
        self.enc_weights_embed = nn.Linear(d_model, 1)
        self.futr_weights_embed = nn.Linear(d_model, 1)


    def forward(self, hist_embed: Tensor, memory: Tensor, out_length: int) -> Tensor:
        hist_embed = hist_embed.permute(0, 2, 1, 3)  # [B,L,N,F]
        memory = memory.permute(0, 2, 1, 3)  # [B,L,N,F]
        bsize, _, obj_len, _ = memory.shape
        query_pos_embed = self._pos_embed(bsize, out_length, obj_len, memory.device)
        tgt_seq = self._tgt_generate(hist_embed, memory, query_pos_embed)
        # b l n f   tgt_seq
        return tgt_seq.permute(0, 2, 1, 3)

    def _pos_embed(self, bsize: int, seq_len: int, obj_len: int, device):
        pos_embed = None
        if self.index_pos_embed:
            idx = torch.arange(seq_len, device=device).reshape(1, seq_len, 1, 1).repeat(bsize, 1, obj_len, 1)
            pos_embed = pos_embed + self.index_pos_embed(idx) if pos_embed else self.index_pos_embed(idx)

        return pos_embed

    def _tgt_generate(self, hist_embed, enc_output, query_pos_embed):
        _, hist_len, obj_len, d_model = hist_embed.shape
        futr_len = query_pos_embed.shape[1]
        futr_weights_embed = self.futr_weights_embed(query_pos_embed).permute(0, 2, 1, 3).reshape(-1, futr_len, 1)
        enc_weights_embed = self.enc_weights_embed(enc_output).permute(0, 2, 1, 3).reshape(-1, hist_len, 1)
        futr_weights = torch.bmm(futr_weights_embed, enc_weights_embed.transpose(-1, -2))
        tgt_seq = torch.bmm(futr_weights, self.tgt_embed(hist_embed).permute(0, 2, 1, 3).reshape(-1, hist_len, d_model))
        tgt_seq = tgt_seq.reshape(-1, obj_len, futr_len, d_model).permute(0, 2, 1, 3)

        return tgt_seq
