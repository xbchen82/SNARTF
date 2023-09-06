import torch
from torch import nn

from model.components.timespa_lib.multiheads import MultiHeads
from .timespa_lib.decoder import Decoder
from .timespa_lib.encoder import GraphEncoder
from .timespa_lib.positional import PositionalEncoding


class TimeSpaJointContextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_type = cfg.network.input_type
        self.context_layer = cfg.network.context_layer
        self.context_head = cfg.network.context_head
        self.attention_dropout = cfg.network.get('context_attention_dropout', 0.)
        self.feature_dropout = cfg.network.get("context_feature_dropout", 0.)
        self.feature_dim = cfg.network.feature_dim
        self.hidden_dim = cfg.network.hidden_dim
        self.context_num = cfg.network.context_num
        self.time_spa_encoder = nn.ModuleList(GraphEncoder(self.context_num, self.feature_dim,
                                                          self.context_head, self.hidden_dim,
                                                          self.attention_dropout)
                                             for _ in range(self.context_layer))

        self.pos_enc = PositionalEncoding(self.feature_dim, self.feature_dropout)

        self.need_enhanced = cfg.get("need_enhanced", True)
        if self.need_enhanced:
            self.feature_enhanced = MultiHeads(feature_dim=self.feature_dim, groups=4, backbone_fc_dim=self.feature_dim)

    def forward(self, tf_in_pos, num_agent):
        action = self.time_spa_encoding(tf_in_pos, num_agent, 0)  # [B,T*N,F]
        for i in range(1, self.context_layer):
            action = self.time_spa_encoding(action, num_agent, i)
        if self.need_enhanced:
            b, t_n, f = action.shape
            action = action.contiguous().view(-1, f)
            _, action, _, _ = self.feature_enhanced(action)
            action = action.contiguous().view(-1, t_n, f)
        return action

    def time_spa_encoding(self, dynamics: torch.Tensor, num_agent: int, num: int) -> torch.Tensor:
        '''
        Params:
            dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features] [B,N,T,F]
        Outputs:
            spatial_dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
        '''
        dynamics = dynamics.contiguous().view(dynamics.shape[0], -1, num_agent, dynamics.shape[-1])  # [B,T,N,F]
        b, t, n, f = dynamics.shape
        dynamics = dynamics.transpose(1, 2).contiguous().view(b * n, t, f)  # [B*N,T,F]
        dynamics = self.pos_enc(dynamics)  # [B*N, T, C]
        time_spa_inputs = dynamics.contiguous().view(b, n, t, f).transpose(1, 2).contiguous().view(b,t*n,f)# [B,T*N,F]
        time_spa_dynamics = self.time_spa_encoder[num](time_spa_inputs, num_agent,
                                                     None,
                                                     None)  # [B,T*N,F]
        return time_spa_dynamics




class TimeSpaJointFutureDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_dim = cfg.network.feature_dim
        self.hidden_dim = cfg.network.hidden_dim
        self.input_type = cfg.network.input_type
        self.future_layer = cfg.network.future_layer
        self.future_head = cfg.network.future_head
        self.attention_dropout = cfg.network.get('future_attention_dropout', 0.)
        self.feature_dropout = cfg.network.get("future_feature_dropout", 0.)
        self.future_num = cfg.network.future_num

        self.time_spa_encoder = nn.ModuleList(GraphEncoder(self.future_num, self.feature_dim,
                                                          self.future_head, self.hidden_dim,
                                                          self.attention_dropout)
                                             for _ in range(self.future_layer))

        self.pos_enc = PositionalEncoding(self.future_num, self.feature_dropout)


        self.decode_num = cfg.network.decode_num
        self.decode_head = cfg.network.decode_head

        self.decoder = Decoder(self.decode_num, self.feature_dim, self.decode_head, self.hidden_dim,
                               self.attention_dropout)

    def forward(self, data, tf_in, num_agent):
        query = tf_in.permute(1, 0, 2)  # [B,T*N,F]
        memory = data.permute(1, 0, 2)  # [B,T*N,F]
        query = self.timesp_encoding(query, num_agent, 0)  # [B,T*N,F]
        for i in range(1, self.future_layer):
            query = self.timesp_encoding(query, num_agent, i)
        query = query.contiguous().view(query.shape[0], -1, num_agent,
                                        query.shape[-1]).transpose(1, 2).contiguous().view(
            query.shape[0] * num_agent, -1, query.shape[-1])
        memory = memory.contiguous().view(memory.shape[0], -1, num_agent,
                                          query.shape[-1]).transpose(1, 2).contiguous().view(
            memory.shape[0] * num_agent, -1, memory.shape[-1])
        output = self.decoder(memory, query, None, None)  # [B, T', C] or [B*N, T', C] if network.multi_agents
        temporal_output = output.contiguous().view(-1, num_agent, output.shape[-2],
                                                   output.shape[-1])  # [B,N,T,C]
        temporal_output = temporal_output.permute(2, 1, 0, 3).contiguous()  # [T,N,B,C]
        temporal_output = temporal_output.view(-1, temporal_output.shape[-2], temporal_output.shape[-1])  # [T*N,B,C]
        return temporal_output  # [T*N,B,C]

    def timesp_encoding(self, dynamics: torch.Tensor, num_agent: int, num: int) -> torch.Tensor:
        '''
        Params:
            dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features] [B,N,T,F]
        Outputs:
            spatial_dynamics: 4d Tensor, shape = [Batch_size, Num_max_agents, Obs_len, Embedded_features]
        '''
        spatial_inputs = dynamics  # [B,T*N,F]
        spatial_dynamics = self.time_spa_encoder[num](spatial_inputs, num_agent,
                                                     None,
                                                     None)  # [B,T*N,F]
        return spatial_dynamics

