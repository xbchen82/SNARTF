from typing import List
import warnings

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupDecayLR(_LRScheduler):
    '''A common learning rate scheduler with warmup and learning rate decay strategy'''
    def __init__(self, optimizer: Optimizer, cfg,
                 last_epoch: int = -1):

        self.warmup_steps = cfg.warmup_steps
        self.warmup_factor = cfg.warmup_factor
        self.warmup_method = cfg.warmup_method

        self.standup_steps = cfg.standup_steps
        self.decay_method = cfg.decay_method

        if not (self.warmup_method in ("constant", "linear", "None")):
            raise ValueError(f"excepted 'warmup_method' be 'linear' or 'constant', "
                             f"got {self.warmup_method}, "
                             f"if you do not want to execute warmup, "
                             f"pass 'None' in configuration file")
        if self.warmup_steps < 0:
            raise ValueError(f"excepted argument 'warmup_steps' must be positive")
        if self.warmup_factor < 0:
            raise ValueError(f"excepted argument 'warmup_factor' must be positive")
        if self.standup_steps < 0:
            raise ValueError(f"excepted argument 'standup_steps' must be positive")

        super(WarmupDecayLR, self).__init__(optimizer, -1)
        if cfg.decay_method == 'None':
            self.learning_rate_decay = None
        elif getattr(torch.optim.lr_scheduler, cfg.decay_method, None) is None:
            if getattr(torch.optim.lr_scheduler, cfg.decay_method+'LR', None) is None:
                raise ValueError(f"cannot find specified learning rate decay method {cfg.decay_method} "
                                 f"in module 'torch.optim.lr_scheduler'")
            else:
                cfg.decay_method += 'LR'
        if cfg.decay_method != 'None':
            from .tools import list2dict
            kwargs = list2dict(cfg.decay_kwargs)
            self.learning_rate_decay = getattr(torch.optim.lr_scheduler, cfg.decay_method)(
                                                    self.optimizer, last_epoch=-1, **kwargs)

        # resume steps
        self.last_epoch = last_epoch
        self._step_count = last_epoch
        self.optimizer._step_count = last_epoch + 1
        if self.last_epoch > self.warmup_steps + self.standup_steps and \
                self.learning_rate_decay is not None:
            for i in range(self.last_epoch - self.warmup_steps - self.standup_steps):
                self.learning_rate_decay.step()

        # set default self._last_lr
        self._get_lr_called_within_step = True
        values = self.get_lr()
        self._get_lr_called_within_step = False
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_steps:
            warmup_factor = 1
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_steps
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [ base_lr * warmup_factor for base_lr in self.base_lrs ]

        elif self.last_epoch <= self.warmup_steps + self.standup_steps or self.learning_rate_decay is None:
            return [ base_lr for base_lr in self.base_lrs]

        else:
            return self.learning_rate_decay.get_last_lr()

    def get_last_lr_factor(self):
        return self.get_last_lr()[0] / self.optimizer.param_groups[0]['initial_lr']

    def step(self):

        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/1.7.1/optim.html#how-to-adjust-learning-rate", UserWarning)
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/1.7.1/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        self._get_lr_called_within_step = True
        self.last_epoch += 1
        if self.last_epoch > self.warmup_steps + self.standup_steps and\
                self.learning_rate_decay is not None:
            self.learning_rate_decay.step()
        values = self.get_lr()
        self._get_lr_called_within_step = False

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
