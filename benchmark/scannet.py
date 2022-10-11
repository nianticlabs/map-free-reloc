import argparse

import numpy as np
import torch
from tqdm import tqdm

from config.default import cfg
from lib.utils.logger import set_log
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from lib.utils.metrics import MetricsAccumulator, print_auc_table, pose_error_torch, A_metrics, precision


def main(args):
    cfg.merge_from_file('config/scannet.yaml')
    cfg.merge_from_file(args.config)

    # Set-up dataloader and model
    datamodule = DataModule(cfg)
    dataset_loader = datamodule.test_dataloader()
    model = build_model(cfg, args.checkpoint)

    # Create logger and save to file
    config_name = args.config.split('/')[-1][:-5]
    set_log(f'results/scannet/{config_name}.txt')

    macc = MetricsAccumulator()

    for data in tqdm(dataset_loader):
        data = data_to_model_device(data, model)
        with torch.no_grad():
            R, t = model(data)
        metrics = pose_error_torch(R, t, data['T_0to1'])
        macc.accumulate(metrics)

    agg_metrics = macc.aggregate()
    print(f"Median Rotation error [deg]: {np.nanmedian(agg_metrics['R_err']):.2f}")
    print(f"Median Translation angular error [deg]: {np.nanmedian(agg_metrics['t_err_ang']):.2f}")
    print(f"Median Translation Euclidean error [m]: {np.nanmedian(agg_metrics['t_err_euc']):.2f}")
    print_auc_table(agg_metrics)

    # compute precision
    thresholds = ((0.1, 5), (0.25, 5), (0.5, 10), (1, 20))
    print("Recall @ "+"/".join([f"({t[0]:.1f}m,{t[1]:.0f}deg)" for t in thresholds])+': '+"/".join(
        ['{:.2f}'.format(precision(agg_metrics, t[1], t[0])) for t in thresholds]))

    # compute A1/A2/A3 metric for translation scale
    a1, a2, a3 = A_metrics(agg_metrics['t_err_scale_sym'])
    print(f"t_scale_error A1/A2/A3 [%]: {a1*100:.1f}/{a2*100:.1f}/{a3*100:.1f}")

    # compute ratio of failures (baselines)
    ratio_failures = np.isnan(agg_metrics['R_err']).mean()
    print(f'failures (not enough corr.) [%]: {ratio_failures*100:.1f}')

    # Save results to `results/' with the name of the config
    np.savez(f'results/scannet/{config_name}', **agg_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--checkpoint', help='path to checkpoint', default='')
    args = parser.parse_args()

    main(args)
