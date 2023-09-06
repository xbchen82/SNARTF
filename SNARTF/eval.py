import os
import logzero, logging
from logzero import logger
import numpy as np
import argparse

from model.utils.eval_tools import AverageMeter
from model.utils.save_tools import load_list_from_folder, isfolder, isfile, find_unique_common_from_lists

""" Metrics """


def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist.mean(axis=-1)  # samples
        ade += dist.min(axis=0)  # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist[..., -1]  # samples
        fde += dist.min(axis=0)  # (1, )
    fde /= len(pred_arr)
    return fde


def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new


def load_data(data_file):
    if isfile(data_file):
        all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float32')  # (frames x agents) x 4
        all_traj = np.expand_dims(all_traj, axis=0)  # 1 x (frames x agents) x 4
    elif isfolder(data_file):
        sample_list, _ = load_list_from_folder(data_file)
        sample_all = []
        for sample in sample_list:
            sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample_all.append(sample)
        all_traj = np.stack(sample_all, axis=0)  # samples x (framex x agents) x 4
    else:
        assert False, 'error'
    return all_traj


def eval_result(results_dir, log):
    gt_dir = f'{os.path.dirname(results_dir)}/gt'
    seq_eval = os.listdir(gt_dir)
    log('loading results from %s' % results_dir)
    log('loading GT from %s' % gt_dir)

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval))
    for seq_name in seq_eval:
        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))
        for data_file in data_filelist:  # each example e.g., seq_0001 - frame_000009
            pred_data = load_data(data_file)
            if "samples" in data_file:
                data_file = data_file.replace("samples", "gt")
                data_file += '.txt'
            else:
                data_file = data_file.replace("recon", "gt")
            gt_data = load_data(data_file)
            # align pred and gt
            id_list = np.unique(pred_data[:, :, 1])
            frame_list = np.unique(pred_data[:, :, 0])
            agent_traj = []
            gt_traj = []
            for idx in id_list:
                # GT traj
                gt_idx = gt_data[gt_data[:, :, 1] == idx]  # frames x 4
                # predicted traj
                ind = np.unique(np.where(pred_data[:, :, 1] == idx)[1].tolist())
                pred_idx = pred_data[:, ind, :]  # sample x frames x 4
                # filter data
                pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
                # append
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)
            """compute stats"""
            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                value = func(agent_traj, gt_traj)
                meter.update(value, n=len(agent_traj))

            stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            log(
                f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}')

    log('-' * 30 + ' STATS ' + '-' * 30)
    for name, meter in stats_meter.items():
        log(f'{meter.count} {name}: {meter.avg:.4f}')
    log('-' * 60)


if __name__ == '__main__':
    results_dir = ''
    logzero.logfile(f"outputs/baseline/eval.log")
    logzero.loglevel(level=logging.INFO)
    eval_result(results_dir, logger.info)
