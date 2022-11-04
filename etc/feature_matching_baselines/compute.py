import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils import parse_7scenes_matching_pairs, parse_mapfree_query_frames, stack_pts, load_scannet_imgpaths
from matchers import LoFTR_matcher, SuperGlue_matcher, SIFT_matcher

MATCHERS = {'LoFTR': LoFTR_matcher, 'SG': SuperGlue_matcher, 'SIFT': SIFT_matcher}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, default='7Scenes',
                        choices=['Scannet', '7Scenes', 'Mapfree'])
    parser.add_argument('--matcher', '-m', type=str, default='SIFT',
                        choices=MATCHERS.keys())
    parser.add_argument('--scenes', '-sc', type=str, nargs='*', default=None)
    parser.add_argument('--pair_txt', type=str,
                        default='test_pairs.5nn.5cm10m.vlad.minmax.txt')  # 7Scenes
    parser.add_argument('--pair_npz', type=str,
                        default='../../data/scannet_indices/scene_data/test/test.npz')  # Scannet
    parser.add_argument('--outdoor', action='store_true',
                        help='use outdoor SG/LoFTR model. If not specified, use indoor models')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == '7Scenes':
        args.data_root = '../../data/sevenscenes'
        scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
        args.scenes = scenes if not args.scenes else args.scenes
        resize = 640, 480
    elif dataset == 'Scannet':
        args.data_root = '../../data/scannet/scans_test'
        resize = 640, 480
    elif dataset == 'Mapfree':
        args.data_root = Path('../../data/mapfree/')
        test_scenes = [folder for folder in (args.data_root / 'test').iterdir() if folder.is_dir()]
        val_scenes = [folder for folder in (args.data_root / 'val').iterdir() if folder.is_dir()]
        args.scenes = test_scenes + val_scenes
        resize = 540, 720

    return args, MATCHERS[args.matcher](resize, args.outdoor)


if __name__ == '__main__':
    args, matcher = get_parser()

    if args.dataset == '7Scenes':
        for scene in args.scenes:
            scene_dir = Path(args.data_root) / scene
            im_pairs = parse_7scenes_matching_pairs(
                str(scene_dir / args.pair_txt))  # {(im1, im2) : (q, t, ess_mat)}
            pair_names = list(im_pairs.keys())
            im_pairs_path = [(str(scene_dir / train_im),
                              str(scene_dir / test_im)) for (train_im, test_im) in pair_names]

            pts_stack = list()
            print(f'Started {scene}')
            for pair in tqdm(im_pairs_path):
                pts = matcher.match(pair)
                pts_stack.append(pts)
            pts_stack = stack_pts(pts_stack)
            results = {'correspondences': pts_stack}
            np.savez_compressed(os.path.join(
                scene_dir,
                f'correspondences_{args.matcher}_{args.pair_txt}.npz'),
                **results)
            print(f'Finished {scene}')

    elif args.dataset == 'Mapfree':
        for scene_dir in args.scenes:
            query_frames_paths = parse_mapfree_query_frames(scene_dir / 'poses.txt')
            im_pairs_path = [(str(scene_dir / 'seq0' / 'frame_00000.jpg'),
                              str(scene_dir / qpath)) for qpath in query_frames_paths]

            pts_stack = list()
            print(f'Started {scene_dir.name}')
            for pair in tqdm(im_pairs_path):
                pts = matcher.match(pair)
                pts_stack.append(pts)
            pts_stack = stack_pts(pts_stack)
            results = {'correspondences': pts_stack}
            np.savez_compressed(scene_dir / f'correspondences_{args.matcher}.npz', **results)
            print(f'Finished {scene_dir.name}')

    elif args.dataset == 'Scannet':
        im_pairs_path = load_scannet_imgpaths(args.pair_npz, args.data_root)
        pts_stack = list()
        print(f'Started Scannet')
        for pair in tqdm(im_pairs_path):
            pts = matcher.match(pair)
            pts_stack.append(pts)
        pts_stack = stack_pts(pts_stack)
        results = {'correspondences': pts_stack}
        np.savez_compressed(
            f'../../data/scannet_misc/correspondences_{args.matcher}_scannet_test.npz',
            **results)
        print(f'Finished Scannet')
    else:
        raise NotImplementedError('Invalid dataset')
