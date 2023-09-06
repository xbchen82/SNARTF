import torch
from torch import nn

from model.components.timeSpaJoint import TimeSpaJointContextEncoder


class ContextEncoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.motion_dim = cfg.dataset.motion_dim

        self.feature_dim = cfg.network.feature_dim
        self.input_type = cfg.network.input_type

        in_dim = self.motion_dim * len(self.input_type)
        self.input_fc = nn.Linear(in_dim, self.feature_dim)
        # timespa
        self.time_spa_joint = TimeSpaJointContextEncoder(cfg)

    def forward(self, data):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['pre_motion'])
            elif key == 'vel':
                vel = data['pre_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['pre_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['pre_motion_scene_norm'])
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.feature_dim)
        time_spa_output = self.time_spa_joint(tf_in.permute(1, 0, 2), data['agent_num'])  # [B,T*N,F]
        data['context_enc'] = time_spa_output.permute(1, 0, 2)
