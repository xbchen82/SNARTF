from math import sqrt

import torch
import torch.nn as nn

class LinearEmbedding(nn.Module):
    '''Implemented embedding layer via nn.Linear with scaling'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(LinearEmbedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) / sqrt(self.out_features)
