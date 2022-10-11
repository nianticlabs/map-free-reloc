import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from config.default import cfg
from lib.utils.logger import set_log
from lib.utils.visualisation import save_video
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from lib.utils.localize import *


def predict(loader, model):
    results_dict = {}

    for data in tqdm(loader):
        # run inference
        data = data_to_model_device(data, model)
        with torch.no_grad():
            R, t = model(data)

        # populate results_dict
        train, test = data['pair_names'][0][0], data['pair_names'][1][0]
        scene = data['scene_id'][0]
        if scene not in results_dict:
            results_dict[scene] = {}
            results_dict[scene]['pair_data'] = {}
            results_dict[scene]['no_pt_pairs'] = []

        if test not in results_dict[scene]['pair_data']:
            results_dict[scene]['pair_data'][test] = {}
            results_dict[scene]['pair_data'][test]['test_pairs'] = []

        # Wrap pose label with RelaPose, AbsPose objects
        train_c, train_q = data['abs_c_0'][0].cpu().numpy(
        ).copy(), data['abs_q_0'][0].cpu().numpy().copy()
        train_abs_pose = AbsPose(train_q, train_c)

        test_c, test_q = data['abs_c_1'][0].cpu().numpy(
        ).copy(), data['abs_q_1'][0].cpu().numpy().copy()
        test_abs_pose = AbsPose(test_q, test_c)
        results_dict[scene]['pair_data'][test]['test_abs_pose'] = test_abs_pose

        rel_t_gt = data['T_0to1'][:, :3, -1].reshape(-1).cpu().numpy().copy()
        rel_q_gt = mat2quat(data['T_0to1'][:, :3, :3].cpu().numpy()).reshape(-1)
        rela_pose_lbl = RelaPose(rel_q_gt, rel_t_gt)

        # check for NaN's in output, meaning failure due to lack of correspondences (for correspondence based methods)
        R = R.detach().cpu().numpy()
        t = t.reshape(-1).detach().cpu().numpy()
        if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
            results_dict[scene]['no_pt_pairs'].append(data['pair_names'])
        else:
            rel_t_pred = t
            rel_q_pred = mat2quat(R).reshape(-1)
            rela_pose_pred = RelaPose(rel_q_pred, rel_t_pred)
            test_pair = RelaPosePair(test, train_abs_pose, rela_pose_lbl,
                                     rela_pose_pred, data['sim'].item())
            test_pair.inliers = data['inliers'] if 'inliers' in data.keys() else 0
            results_dict[scene]['pair_data'][test]['test_pairs'].append(test_pair)

    return results_dict


def eval(args):
    # Load configs
    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    # update test pair txt from arguments (can be set at dataset config)
    if args.test_pair_txt:
        cfg.DATASET.PAIRS_TXT.TEST = args.test_pair_txt
    if args.one_nn:
        cfg.DATASET.PAIRS_TXT.ONE_NN = True

    # Set log object
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_log(args.output_root / 'test_results.txt')

    # Create dataloader
    dataloader = DataModule(cfg).test_dataloader()

    # Create model
    model = build_model(cfg, args.checkpoint)

    # Get predictions from model
    results_dict = predict(dataloader, model)
    np.save(args.output_root / 'rawpred.npy', results_dict)  # save, just in case

    # Evaluate
    err_thres = ((0.1, 5), (0.25, 5), (0.5, 10), (1, 20))  # (meters, deg)
    save_res_path = args.output_root / 'results.npy'
    if args.triang:
        # Using triangulation + RANSAC
        eval_pipeline_with_ransac(results_dict, None, ransac_thres=args.triang_ransac_thres,
                                  ransac_iter=10, ransac_miu=1.414, pair_type='relapose',
                                  err_thres=err_thres, save_res_path=save_res_path)
    else:
        # Directly using metric relative pose estimate to obtain absolute query pose
        # NOTE: if there are more than 1NN for a query, the absolute pose is obtained by
        # the geometric median of absolute translation vectors of each NN, and
        # L2 chordal mean rotation of abs. rotation matrices of each NN (see more details in cal_abs_pose_err_metric)
        eval_pipeline_without_ransac(results_dict, err_thres=err_thres, save_res_path=save_res_path)

    # Create txt file per scene showing predicted pose of each query
    save_results_visualisation(save_res_path)

    # Create precision/recall plots
    generate_precision_recall_plots(save_res_path, err_thres[1])

    if args.save_video:
        save_video(save_res_path, dataloader, args.output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('dataset_config', help='path to dataset config file')
    parser.add_argument('--checkpoint', help='path to model checkpoint', default='')
    parser.add_argument('--test_pair_txt', '-pair', type=str, default=None)
    parser.add_argument('--output_root', '-odir', type=str, default='results/')
    parser.add_argument(
        '--one_nn', action='store_true',
        help='keep only one nearest neighbour, the one with highest VLAD similarity. Applicable for 7Scenes, which has more than one NN. No effect on MapFree dataset, which by definition only contains 1 keyframe per scene.')
    parser.add_argument(
        '--triang', action='store_true',
        help='uses triangulation to compute absolute pose of query image. Only applicable for 7Scenes.')
    parser.add_argument(
        '--triang_ransac_thres', '-rthres', metavar='%d', type=int, nargs='+', default=[15],
        help='the set of triangulation ransac inlier thresolds(angle error)(default: %(default)s)')
    parser.add_argument(
        '--save_video', action='store_true',
        help='create a video per sequence showing results per frame (valid only for 1NN cases)')

    args = parser.parse_args()
    args.output_root = Path(args.output_root)
    assert (args.one_nn and args.triang) != True, 'triangulation needs more than one nearest neighbour'
    if args.save_video:
        assert args.one_nn, 'video option only available when using a single keyframe (1 nearest neighbour)'

    eval(args)
