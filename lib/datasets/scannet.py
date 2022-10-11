# Based on https://github.com/zju3dv/LoFTR/blob/master/src/datasets/scannet.py
from os import path as osp
from os import listdir

import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv

from lib.datasets.utils import (
    read_color_image,
    read_depth_image,
    read_scannet_pose,
    read_scannet_intrinsic,
    correct_intrinsic_scale
)


class ScanNetScene(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 resize=(640, 480),
                 estimated_depth=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.resize = resize

        # prepare data_names, intrinsics and extrinsics(T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]

        # for training
        self.augment_fn = augment_fn if mode == 'train' else None

        # load pre-computed estimated depth, if exists
        self.depthmaps = np.load(estimated_depth) if estimated_depth is not None else None

    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.root_dir,
                       scene_name,
                       'sensor_data', f'frame-{name:06}.pose.txt')
        return read_scannet_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)

        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = self.data_names[idx]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # loads image and rescales. apply augmentation if available
        img_name0 = osp.join(self.root_dir, scene_name, 'sensor_data',
                             f'frame-{stem_name_0:06}.color.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'sensor_data',
                             f'frame-{stem_name_1:06}.color.jpg')
        image0 = read_color_image(img_name0, resize=self.resize, augment_fn=self.augment_fn)
        image1 = read_color_image(img_name1, resize=self.resize, augment_fn=self.augment_fn)

        # read the depthmap which is stored as (480, 640)
        if self.mode in ['test']:
            if self.depthmaps is None:
                # Load GT depth
                dimg_name0 = osp.join(self.root_dir, scene_name, 'sensor_data',
                                      f'frame-{stem_name_0:06}.depth.pgm')
                dimg_name1 = osp.join(self.root_dir, scene_name, 'sensor_data',
                                      f'frame-{stem_name_1:06}.depth.pgm')
                depth0 = read_depth_image(dimg_name0)
                depth1 = read_depth_image(dimg_name1)
            else:
                # Load pre-computed depth (using arbitrary methods) from npz file
                def key(frame_idx): return f'{scene_name[5:]}_frame_{frame_idx:06}'
                depth0 = torch.from_numpy(self.depthmaps[key(stem_name_0)].astype(np.float32))
                depth1 = torch.from_numpy(self.depthmaps[key(stem_name_1)].astype(np.float32))
        else:
            depth0 = depth1 = torch.tensor([])

        # get intrinsics
        intrinsics_path = osp.join(self.root_dir, scene_name, 'sensor_data', '_info.txt')
        K_color = read_scannet_intrinsic(intrinsics_path, color=True)
        K_color = correct_intrinsic_scale(
            K_color, scale_x=self.resize[0] / 1296, scale_y=self.resize[1] / 968)
        K_color = torch.from_numpy(K_color)
        K_depth = torch.from_numpy(read_scannet_intrinsic(intrinsics_path, color=False))

        # read and compute relative poses
        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (3, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K_color0': K_color,  # (3, 3)
            'K_color1': K_color,  # (3, 3)
            'K_depth': K_depth,  # (3, 3)
            'dataset_name': 'ScanNet',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg'))
        }

        return data


class ScanNetDataset(utils.data.ConcatDataset):
    def __init__(self,
                 cfg,
                 mode: str,
                 transforms=None):
        assert mode in ('train', 'val', 'test'), 'Invalid dataset mode'

        root_dir = cfg.DATASET.DATA_ROOT
        index_npz_dir = cfg.DATASET.NPZ_ROOT
        min_overlap_score = cfg.DATASET.MIN_OVERLAP_SCORE
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH

        # create a dataset for each npz file
        # usually each npz file contains the information for a single scene (training and val)
        # however, for testing all pairs are concatenated into a single npz file (test.npz)
        root_dir = osp.join(root_dir, 'scans_test' if mode == 'test' else 'scans')
        npz_path = osp.join(index_npz_dir, mode)
        npz_list = [osp.join(npz_path, fname) for fname in listdir(npz_path) if fname[-3:] == 'npz']

        dataset_list = [ScanNetScene(root_dir=root_dir,
                                     npz_path=npz_fname,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     augment_fn=transforms,
                                     resize=resize,
                                     estimated_depth=estimated_depth) for npz_fname in npz_list]

        super().__init__(dataset_list)
