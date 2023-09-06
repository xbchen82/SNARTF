from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中代码实现
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class FeedForwardNet(nn.Module):

    def __init__(self, inputs: int, hidden: int, dropout: Optional[float]):
        '''Implemented feedforward network'''
        super(FeedForwardNet, self).__init__()
        self.inputs = inputs
        self.hidden = hidden

        self.upscale = nn.Linear(inputs, hidden)
        self.activation = Mish()
        self.downscale = nn.Linear(hidden, inputs)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upscale(x)
        x = self.activation(x)
        x = self.downscale(x)
        return self.dropout(x)
