from collections import defaultdict

import numpy as np
import torch
from torch import nn

from model.components.contextEncoder import ContextEncoder
from model.components.futureDecoder import FutureDecoder
from model.loss.loss import loss_func


class SNARTF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.ar_train = cfg.get('ar_train', False)
        self.loss_cfg = cfg.training.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.data = None
        self.context_encoder = ContextEncoder(cfg)
        self.future_decoder = FutureDecoder(cfg)

    def set_data(self, data):
        self.data = data

    def forward(self):
        self.context_encoder(self.data)
        self.inference(sample_num=self.loss_cfg['sample']['k'])
        return self.data

    def inference(self, sample_num=20):
        if self.data['context_enc'] is None:
            self.context_encoder(self.data)
        self.future_decoder(self.data, sample_num=sample_num)
        return self.data['dec_motion'], self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict
