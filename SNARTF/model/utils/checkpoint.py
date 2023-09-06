from pathlib import Path
from typing import Union, List

import torch
import re, os


def load_checkpoint_for_evaluation(model, output_dir: Union[str, Path],
                                   device: torch.device = "cpu",
                                   epoch: int = 0) -> int:
    '''Helper function for loading saved model weights'''
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    pths = os.listdir(output_dir)
    epochs = []
    for name in pths:
        if name.startswith('model') and name.endswith('.pth'):
            epochs.append(re.findall(rf'([0-9]+)', name)[0])

    return epoch + 1


def load_checkpoint(model, optimizer, scheduler, output_dir: Union[str, Path],
                    device: torch.device = "cpu",
                    epoch: int = 0) -> int:
    '''Helper function for loading saved model weights'''
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    pths = os.listdir(output_dir)
    epochs = []
    for name in pths:
        if name.startswith('model') and name.endswith('.pth'):
            epochs.append(re.findall(rf'([0-9]+)', name)[0])
    if epochs:
        epochs = list(map(int, epochs))
        if epoch == 0:  # load last checkpoint
            epoch = max(epochs)
        elif epoch > 0:  # load {epoch}th checkpoint
            if not (epoch in epochs):
                raise FileNotFoundError
        else:  # load {-epoch}th checkpoint from the end
            if len(epochs) < -epoch:
                raise IndexError
            else:
                epoch = epochs[epoch]
    else:
        return 1
    path = f'{output_dir}/model_{epoch}.pth'
    model_dict = torch.load(path, map_location=device)

    model.load_state_dict(model_dict['model_dict'])
    if 'opt_dict' in model_dict:
        optimizer.load_state_dict(model_dict['opt_dict'])
        if device != 'cpu':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    if 'scheduler_dict' in model_dict:
        scheduler.load_state_dict(model_dict['scheduler_dict'])
    return epoch + 1


def save_checkpoint(epoch: int,
                    output_dir: Union[str, Path], model, optimizer, scheduler) -> None:
    '''Helper function for saving trained model weights'''
    model = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
             'scheduler_dict': scheduler.state_dict(), 'epoch': epoch}

    torch.save(model, f'{output_dir}/model_{epoch}.pth')
