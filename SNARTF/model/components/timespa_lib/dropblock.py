import torch
import torch.nn as nn

class DropBlock(nn.Module):
    '''Implemented dropout block for input data'''
    def __init__(self, dropout_rate, padding_value=0., channel_first=False):
        super(DropBlock, self).__init__()
        assert dropout_rate > 0 and dropout_rate < 1., 'invalid dropout rate'
        self.dropout_rate = dropout_rate
        self.padding_value = padding_value
        assert channel_first == True or channel_first == False, 'channel_first must be a boolean flag'
        self.channel_first = channel_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_first:
            x = x.transpose(1, -1)
        drop = torch.rand(x.shape[:-1], device=x.device)
        drop_ind = drop < self.dropout_rate
        x[drop_ind] = self.padding_value
        if self.channel_first:
            x = x.transpose(1, -1)
        return x
