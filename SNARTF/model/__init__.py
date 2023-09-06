from typing import Union, Any, List

from torch.utils.data import DataLoader

from .data.eth_ucy.eth_ucy_dataloader import eth_ucy_data_dataloader
from .data.sdd.sdd_dataloader import sdd_data_dataloader
from .snartf import SNARTF
from .utils.tools import make_optimizer, make_scheduler


def make_dataset(cfg, device, logger, split='train', phase='training'):
    """Return a list of datasets splited into train, valid, and test according to given configuration"""
    if cfg.name.lower() in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
        dataloader = eth_ucy_data_dataloader
    elif cfg.name.lower() in 'sdd':
        dataloader = sdd_data_dataloader
    else:
        raise NotImplementedError
    return dataloader(cfg, device, logger, split, phase)


def make_dataloader(*datasets, num_workers: int = 8):
    """Initialize a list of dataloaders according to given datasets"""

    def _worker_init_fn(worker_id):
        import torch, numpy, random
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    from torch.utils.data import DataLoader
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset, batch_size=1, num_workers=num_workers,
                                      shuffle=True, pin_memory=True))
    if len(dataloaders) == 1:
        return dataloaders[0]
    else:
        return dataloaders
