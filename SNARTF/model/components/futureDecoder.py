import torch
from torch import nn

from model.components.common.dist import Normal, Categorical
from model.components.common.utils import initialize_weights, MLP
from model.components.timeSpaJoint import TimeSpaJointFutureDecoder
from model.components.timespa_lib.nonArQgb import NonAutoRegression


class PositionalAgentEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False,
                 agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)


class FutureDecoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.future_frames = cfg.dataset.future_frames
        self.forecast_dim = cfg.dataset.forecast_dim

        self.feature_dim = cfg.network.feature_dim
        self.pred_type = cfg.network.pred_type
        self.input_type = cfg.network.input_type
        self.out_mlp_dim = cfg.network.out_mlp_dim

        self.nz = cfg.network.nz
        self.z_type = cfg.network.z_type

        in_dim = len(self.input_type) * self.forecast_dim
        self.input_fc = nn.Linear(in_dim, self.feature_dim)
        self.true_query_fc = nn.Linear(in_dim, self.feature_dim)
        self.noise_fc = nn.Linear(self.nz, self.feature_dim)

        self.QGB = NonAutoRegression(feature_dims=self.feature_dim,
                                     pred_seq_len=self.future_frames, d_model=self.feature_dim)

        self.time_spa_decoder = TimeSpaJointFutureDecoder(cfg)

        if self.out_mlp_dim == 0:
            self.out_fc = nn.Linear(self.feature_dim, self.forecast_dim)
        else:
            in_dim = self.feature_dim
            self.out_mlp = MLP(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, self.forecast_dim)
        initialize_weights(self.out_fc.modules())

    def decode_traj_batch(self, data, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        agent_num = data['agent_num']
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(pre_motion)
            elif key == 'vel':
                vel = pre_vel
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['pre_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(pre_motion_scene_norm)
            else:
                raise ValueError('unknown input_type!')

        traj_in = torch.cat(traj_in, dim=-1)
        dec_in = traj_in.view(-1, sample_num, traj_in.shape[-1])
        tf_in = self.input_fc(dec_in.view(-1, dec_in.shape[-1])).view(dec_in.shape[0], -1, self.feature_dim)
        tf_in = tf_in.view(-1, agent_num, tf_in.shape[-2], tf_in.shape[-1])  # [T,N,B,F]
        tf_in = tf_in.permute(2, 1, 0, 3)  # [B,N,T,F]
        context_qgb = context.view(-1, agent_num, context.shape[-2], context.shape[-1])  # [T,N,B,F]
        context_qgb = context_qgb.permute(2, 1, 0, 3)
        query = self.QGB(tf_in, context_qgb, self.future_frames)  # [B,N,T,F]

        z_in = z.view(-1, sample_num, z.shape[-1])
        z_in = z_in.repeat(self.future_frames, 1, 1)
        query_z_in = z_in.view(-1, agent_num, z_in.shape[-2], z_in.shape[-1])
        query_z_in = self.noise_fc(query_z_in)

        query = (query.permute(2, 1, 0, 3) + query_z_in).contiguous().view(-1, query_z_in.shape[-2],
                                                                           query_z_in.shape[-1])
        tf_out = self.time_spa_decoder(context, query, num_agent=data['agent_num'])  # [T*N,B,C]

        out_tmp = tf_out.contiguous().view(-1, tf_out.shape[-1])

        if self.out_mlp_dim != 0:
            out_tmp = self.out_mlp(out_tmp)
        seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
        if self.pred_type in {'vel', 'scene_norm'}:
            norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])  # [T,N*B,C]
            if self.pred_type == 'vel':
                norm_motion = torch.cumsum(norm_motion, dim=0)
            seq_out = norm_motion + pre_motion_scene_norm[[-1]]
            seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])  # [T*N,B,C]
        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])  # [T,N*B,C]
        data['seq_out'] = seq_out

        if self.pred_type == 'vel':
            dec_motion = seq_out + pre_motion[[-1]]
        elif self.pred_type == 'pos':
            dec_motion = seq_out.clone()
        elif self.pred_type == 'scene_norm':
            dec_motion = seq_out + data['scene_orig']
        else:
            dec_motion = seq_out + pre_motion[[-1]]

        dec_motion = dec_motion.transpose(0, 1).contiguous()  # [N*S,T,C]
        dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])  # [N,S,T,C]
        data['dec_motion'] = dec_motion

    def forward(self, data, sample_num=1, autoregress=False):
        context = data['context_enc'].repeat_interleave(sample_num, dim=1)  # 80 x 64
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=1)  # 10 x 80 x 2
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=1)  # if self.pred_type == 'vel' else None
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=1)
        if self.z_type == 'gaussian':
            data['p_z_dist'] = Normal(mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device),
                                      logvar=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))
        else:
            raise ValueError('unknown z_type!')
        z = data['p_z_dist'].sample()
        self.decode_traj_batch(data, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)
