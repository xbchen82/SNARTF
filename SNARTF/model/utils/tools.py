from typing import List
import numpy as np
import random, os, shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler


def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None, decay_gamma=0.1):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=decay_gamma)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def list2dict(_list: List[str]) -> dict:
    if not isinstance(_list, List):
        raise TypeError(f"excepted input as List, got {type(_list)} instead")
    if len(_list) % 2 != 0:
        raise ValueError(f"input decay_kwargs has odd length {len(_list)}, "
                         f"excepted input list must be a list of <key, value> pairs")
    _dict = {}
    for i in range(0, len(_list), 2):
        if not isinstance(_list[i], str):
            raise ValueError(f"excepted input in list[{i}] as a keyword, "
                             f"a keyword must be a string, got a {type(_list)}")
        _dict[_list[i]] = _list[i + 1]
    return _dict


def make_optimizer(model: nn.Module, cfg) -> Optimizer:
    '''Build a optimizer according to given model and configs'''
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.lr_scheduler == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif cfg.lr_scheduler == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')
    return optimizer, scheduler


from .scheduler import WarmupDecayLR


def make_scheduler(optimizer: Optimizer, cfg, last_epoch: int = -1) -> WarmupDecayLR:
    '''Build a learning rate scheduler with warmup and learning rate decay strategy'''
    WDLR = WarmupDecayLR(optimizer, cfg, last_epoch)
    return WDLR


def set_seed(cfg):
    '''Fix random seeds for reproduction'''
    seed = cfg.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cudnn.benchmark = cfg.benchmark


def backup_file(cfg):
    '''Create mirror of specified files in /model/ at the begining of training'''
    if len(cfg.backup) == 0:
        return

    _BACKUP_DIR = Path(cfg.workspace) / 'backup'
    _SOURCE_DIR = Path("model")

    for file in cfg.backup:
        source = _SOURCE_DIR / file
        target = _BACKUP_DIR / source
        if not os.path.exists(target.parent):
            os.makedirs(target.parent)
        shutil.copy(source, target)




