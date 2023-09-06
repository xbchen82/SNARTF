import os, sys, time
import logging, logzero
import subprocess

from logzero import logger
from tqdm import tqdm
from pathlib import Path
import click
import numpy as np
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
import time

from eval import eval_result
from model.utils.save_tools import mkdir_if_missing, save_prediction

torch.set_printoptions(precision=4, linewidth=400, threshold=sys.maxsize, sci_mode=False)

from model import SNARTF, make_dataloader, make_dataset, make_optimizer, make_scheduler
from model.configs import get_default_configs, get_modified_configs
from model.utils import load_checkpoint, save_checkpoint, backup_file, set_seed
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@click.group()
def clk():
    pass


@clk.command()
@click.option("-c", "--config-file", type=str, default="")
@click.argument("cmd-config", nargs=-1)
def train(config_file, cmd_config):
    cfg = get_modified_configs(config_file, cmd_config)
    set_seed(cfg)

    if not os.path.exists(cfg.workspace):
        os.makedirs(cfg.workspace)
    logzero.logfile(f"{cfg.workspace}/train.log")
    logzero.loglevel(level=logging.INFO)
    logger.info("Starting train process...")
    logger.info(cfg)

    logger.info("Building dataloaders...")
    generator = make_dataset(cfg.dataset, cfg.device, logger.info, split='train', phase='training')

    logger.info("Building model...")
    model = SNARTF(cfg)

    logger.info("Building optimizer and scheduler...")
    optimizer, scheduler = make_optimizer(model, cfg.training)

    logger.info(f"Loading checkpoint from {cfg.workspace}")
    start_epoch = load_checkpoint(model, optimizer, scheduler, cfg.workspace, device=cfg.device, epoch=0)
    if start_epoch > cfg.training.epochs:
        logger.info(f"The training corresponding to config file {Path(config_file).name} was over.")
        return
    elif start_epoch > 1:
        logger.info(f"Loaded checkpoint successfully! Start epoch is {start_epoch}.")
    else:
        logger.info(f"Cannot find pre-trained checkpoint. Start training from epoch 1.")
        backup_file(cfg)
        logger.info(f"Created backup successfully!")
    _NUM_CUDA_DEVICES = 0
    if cfg.device == 'cuda':
        _NUM_CUDA_DEVICES = torch.cuda.device_count()
        if _NUM_CUDA_DEVICES < 1:
            raise ValueError(f"cannot perform cuda training due to insufficient cuda device.")
        logger.info(f"{_NUM_CUDA_DEVICES} cuda device found!")
        model = model.cuda()

    logger.info("Start training!")
    for epoch in range(start_epoch, cfg.training.epochs + 1):

        t0 = time.time()
        data_time = 0
        forward_time = 0
        backward_time = 0
        train_time = 0
        model.train()
        loss_names = list(cfg.training.loss_cfg.keys())
        loss_map = {}
        for key in loss_names:
            loss_map[key] = 0.
        loss_map['total_loss'] = 0.
        # train loop
        for iter, (_, data) in enumerate(generator):
            t1 = time.time()
            data_time += t1 - t0
            if data is None:
                continue
            model.set_data(data)
            model()
            total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
            t2 = time.time()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            t3 = time.time()

            forward_time += t2 - t1
            backward_time += t3 - t2
            train_time += t3 - t0

            for key in loss_unweighted_dict:
                loss_map[key] += loss_unweighted_dict[key]
            loss_map['total_loss'] += total_loss.item()
            if (iter % cfg.training.print_freq) == 0 and iter > 0:
                eta = train_time / iter * (len(generator) - iter)
                progress = iter / len(generator) * 100
                log_str = 'Loss: '
                for key in loss_map:
                    log_str = log_str + f"{key}:{(loss_map[key] / iter):5.4f},"
                logger.info(
                    f"Epoch: {epoch:04d}, Progress: {progress:5.2f} [%], ETA: {eta:8.2f} [s], {iter}/{len(generator)} | "
                    f"{log_str} Learning Rate: {(optimizer.state_dict()['param_groups'][0]['lr']):.3e}")

            if cfg.training.scheduled_by_steps:
                scheduler.step()

            t0 = time.time()

        if not cfg.training.scheduled_by_steps:
            scheduler.step()

        logger.info(f"Epoch {epoch:04d} completed, Time: {train_time:8.2f} [s]. "
                    f"Data time:{data_time:8.2f} [s], Forward time: {forward_time:8.2f} [s], Backward time: {backward_time:8.2f} [s]")
        if epoch % cfg.training.model_save_freq == 0:
            logger.info(f"Saving checkpoint {epoch}")
            save_checkpoint(epoch, cfg.workspace, model=model, optimizer=optimizer, scheduler=scheduler)
        logger.info("-" * 20)


@clk.command()
@click.option("-c", "--config-file", type=str, default="")
@click.argument("cmd-config", nargs=-1)
def eval(config_file, cmd_config):
    cfg = get_modified_configs(config_file, cmd_config)
    set_seed(cfg)
    if not os.path.exists(cfg.workspace):
        raise FileNotFoundError
    logzero.logfile(f"{cfg.workspace}/eval.log")
    logzero.loglevel(level=logging.INFO)
    logger.info("Starting evaluation process...")
    logger.info(cfg)

    logger.info("Building dataloaders...")
    generator = make_dataset(cfg.dataset, cfg.device, logger.info, split=cfg.evaluation.split, phase='training')

    logger.info("Building model...")
    model = SNARTF(cfg)

    logger.info(f"Loading checkpoint from {cfg.workspace}")
    if type(cfg.evaluation.epoch) == int:
        epochs = [cfg.evaluation.epoch]
    else:
        epochs = cfg.evaluation.epoch
    for epoch in epochs:
        eval_epoch(model, generator, cfg, epoch)


def eval_epoch(model, generator, cfg, epoch):
    try:
        model_dict = torch.load(f'{cfg.workspace}/model_{epoch}.pth', map_location=cfg.device)
        model.load_state_dict(model_dict['model_dict'], strict=False)
    except Exception:
        logger.error(f"Failed to load checkpoint at epoch {epoch}.")
        # return
    logger.info(f"Loaded checkpoint at epoch {epoch} successfully!")

    if cfg.device == 'cuda':
        _NUM_CUDA_DEVICES = torch.cuda.device_count()
        if _NUM_CUDA_DEVICES < 1:
            raise ValueError(f"cannot perform cuda training due to insufficient cuda device.")
        logger.info(f"{_NUM_CUDA_DEVICES} cuda device found!")
        model = model.cuda()
    model.eval()
    logger.info("Start evaluation!")
    # evaluation loop
    t0 = time.time()
    model.eval()
    total_num_pred = 0
    with torch.no_grad():
        for iter, (origin_data, data) in enumerate(generator):
            print('\r', f'generate process {iter}/{len(generator)}', end='', flush=True)
            t1 = time.time()
            if data is None:
                continue
            model.set_data(data)
            sample_motion_3D, data = model.inference(sample_num=cfg.evaluation.sample_k)
            gt_motion_3D = data['fut_motion'].transpose(0, 1) * cfg.dataset.traj_scale
            sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous() * cfg.dataset.traj_scale

            """save samples"""
            sample_dir = os.path.join(cfg.workspace, 'samples')
            mkdir_if_missing(sample_dir)
            gt_dir = os.path.join(cfg.workspace, 'gt')
            mkdir_if_missing(gt_dir)

            for i in range(sample_motion_3D.shape[0]):
                save_prediction(sample_motion_3D[i], origin_data, f'/sample_{i:03d}', sample_dir,
                                cfg.dataset.future_frames)
            num_pred = save_prediction(gt_motion_3D, origin_data, '', gt_dir, cfg.dataset.future_frames)  # save gt
            total_num_pred += num_pred
        logger.info(f'\n\n total_num_pred: {total_num_pred}')
        logger.info(f'start evaluate epoch: {epoch}!')
        eval_result(sample_dir, logger.info)


if __name__ == '__main__':
    clk()
