from pathlib import Path
import re

import torch
import torch.utils.data as data
import numpy as np
from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat

from lib.datasets.utils import read_color_image, read_depth_image, correct_intrinsic_scale
from pytorch_lightning import _logger as logger

from lib.utils.rotationutils import relative_pose_wxyz


class MapFreeScene(data.Dataset):
    def __init__(self, scene_root, resize, sample_factor=1, overlap_limits=None, transforms=None,
                 estimated_depth=None, sample_offset: int = 0):
        super().__init__()

        self.scene_root = Path(scene_root)
        self.resize = resize
        self.sample_factor = sample_factor
        self.sample_offset = sample_offset
        self.transforms = transforms
        self.estimated_depth = estimated_depth

        # load absolute poses
        self.poses = self.read_poses(self.scene_root)

        # read intrinsics
        self.K = self.read_intrinsics(self.scene_root, resize)

        # load pairs
        self.pairs = self.load_pairs(scene_root=self.scene_root, overlap_limits=overlap_limits,
                                     sample_factor=self.sample_factor, sample_offset=sample_offset)

    @staticmethod
    def read_intrinsics(scene_root: Path, resize=None):
        Ks = {}
        with (scene_root / 'intrinsics.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                if resize is not None:
                    K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
                Ks[img_name] = K
        return Ks

    @staticmethod
    def read_poses(scene_root: Path, filename: str = 'poses.txt'):
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        poses = {}
        with (scene_root / filename).open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[:4], qt[4:])
        return poses

    def load_pairs(self, scene_root: Path, overlap_limits: tuple = None, sample_factor: int = 1,
                   sample_offset: int = 0):
        """
        For training scenes, filter pairs of frames based on overlap (pre-computed in overlaps.npz)
        For test/val scenes, pairs are formed between keyframe and every other sample_factor query frames.
        If sample_factor == 1, all query frames are used. Note: sample_factor applicable only to test/val
        Returns:
        pairs: np.ndarray [Npairs, 4], where each column represents seaA, imA, seqB, imB, respectively
        """
        overlaps_path = scene_root / 'overlaps.npz'

        if overlaps_path.exists():  # train case
            f = np.load(overlaps_path, allow_pickle=True)
            idxs, overlaps = f['idxs'], f['overlaps']
            if overlap_limits is not None:
                min_overlap, max_overlap = overlap_limits
                mask = (overlaps > min_overlap) * (overlaps < max_overlap)
                idxs = idxs[mask]
            if 1 < sample_offset:
                idxs2 = []
                for seqA, imgA, seqB, imgB in idxs:
                    if seqA == seqB:  # filter cases where the query sequence contains the
                        queries = [imgB_
                                   for i in range(sample_offset)
                                   if (0 <= (imgB_ := imgB - sample_offset + 1 + i)
                                       and imgB_ != imgA)  # don't allow the map frame in the query
                                   ]
                    else:
                        queries = [imgB_
                                   for i in range(sample_offset)
                                   if 0 <= (imgB_ := imgB - sample_offset + 1 + i)]
                    if len(queries) == sample_offset:
                        idxs2.append((seqA, imgA, seqB, tuple(queries)))
                print(f"Filtered {len(idxs) - len(idxs2)} pairs")
                idxs = idxs2
                del idxs2
            else:
                assert sample_factor == 1
            return idxs.copy()
        else:  # val and test case
            idxs = np.zeros((len(self.poses) - 1, 4), dtype=np.uint16)
            idxs[:, 2] = 1
            # match number between '_' and '.' in the filename, e.g. 'seq1/frame_00001.jpg'
            pattern = r'_(\d+)\..*$'
            idxs[:, 3] = np.array([re.search(pattern, fn).group(1)
                                   for fn in self.poses.keys()
                                   if 'seq0' not in fn],
                                  dtype=np.uint16)
            idxs = idxs[sample_offset::sample_factor]
            if 1 < sample_offset:
                idxs = [(seqA, imgA, seqB, tuple(range(imgB - sample_offset + 1, imgB + 1)))
                        for seqA, imgA, seqB, imgB in idxs]
            else:
                assert sample_factor == 5

            # a = np.empty(dtype=np.dtype([('a', (int, 3)), ('b', (int, 9))]))
            # a[:, :3] = idxs[:, :3]
            # for
            # idxs[:, 3] = np.tile(idxs[:, 3:4], (1, sample_offset)) \
            #              + np.tile(np.arange(-sample_offset + 1, 1), (idxs.shape[0], 1))
            return idxs

    def get_pair_path(self, pair):
        seqA, imgA, seqB, imgB = pair
        return (f'seq{seqA}/frame_{imgA:05}.jpg', f'seq{seqB}/frame_{imgB:05}.jpg')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # image paths (relative to scene_root)
        im1_path, im2_path = self.get_pair_path(self.pairs[index])

        # load color images
        image1 = read_color_image(self.scene_root / im1_path,
                                  self.resize, augment_fn=self.transforms)
        image2 = read_color_image(self.scene_root / im2_path,
                                  self.resize, augment_fn=self.transforms)

        # load depth maps
        if self.estimated_depth is not None:
            dim1_path = str(self.scene_root / im1_path).replace('.jpg',
                                                                f'.{self.estimated_depth}.png')
            dim2_path = str(self.scene_root / im2_path).replace('.jpg',
                                                                f'.{self.estimated_depth}.png')
            depth1 = read_depth_image(dim1_path)
            depth2 = read_depth_image(dim2_path)
        else:
            depth1 = depth2 = torch.tensor([])

        # get absolute pose of im0 and im1
        # quaternion and translation vector that transforms World-to-Cam
        q1, t1 = self.poses[im1_path]
        # quaternion and translation vector that transforms World-to-Cam
        q2, t2 = self.poses[im2_path]
        c1 = rotate_vector(-t1, qinverse(q1))  # center of camera 1 in world coordinates
        c2 = rotate_vector(-t2, qinverse(q2))  # center of camera 2 in world coordinates

        # get 4 x 4 relative pose transformation matrix (from im1 to im2)
        # for test/val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
        # q12 = qmult(q2, qinverse(q1))
        # t12 = t2 - rotate_vector(t1, q12)
        q12, t12 = relative_pose_wxyz(q1_wxyz=q1, t1=t1, q2_wxyz=q2, t2=t2)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)

        data = {
            'image0': image1,  # (3, h, w)
            'depth0': depth1,  # (h, w)
            'image1': image2,
            'depth1': depth2,
            'T_0to1': T,  # (4, 4)  # relative pose
            'abs_q_0': q1,
            'abs_c_0': c1,
            'abs_q_1': q2,
            'abs_c_1': c2,
            'K_color0': self.K[im1_path].copy(),  # (3, 3)
            'K_color1': self.K[im2_path].copy(),  # (3, 3)
            'dataset_name': 'Mapfree',
            'scene_id': self.scene_root.stem,
            'scene_root': str(self.scene_root),
            'pair_id': index*self.sample_factor,
            'pair_names': (im1_path, im2_path),
            'sim': 0.  # needed for 7Scenes eval compatibility
        }

        return data


class MapFreeSceneMultiFrame(MapFreeScene):
    def __init__(self, scene_root, resize, sample_factor=1, overlap_limits=None, transforms=None,
                 estimated_depth=None, sample_offset: int = 0):
        super().__init__(scene_root=scene_root,
                         resize=resize,
                         sample_factor=sample_factor,
                         overlap_limits=overlap_limits,
                         transforms=transforms,
                         estimated_depth=estimated_depth,
                         sample_offset=sample_offset)

        # load device tracking poses
        # if 'train' not in str(self.scene_root):
        self.poses_device = self.read_poses(scene_root=self.scene_root, filename='poses_device.txt')
        # else:
        #     self.poses_device = None
        #     print("TODO: put back for train")

    def get_pair_path(self, pair):
        seqA, imgA, seqB, imgB = pair
        return (f'seq{seqA}/frame_{imgA:05}.jpg',
                tuple(f'seq{seqB}/frame_{imgB_:05}.jpg' for imgB_ in imgB))

    def __getitem__(self, index):
        # image paths (relative to scene_root)
        im1_path, im2_path = self.get_pair_path(self.pairs[index])

        # load color images
        image1 = read_color_image(path=self.scene_root / im1_path,
                                  resize=self.resize, augment_fn=self.transforms)
        image2 = torch.stack([read_color_image(path=self.scene_root / im2_path_,
                                               resize=self.resize, augment_fn=self.transforms)
                              for im2_path_ in im2_path])

        # load depth maps
        if self.estimated_depth is not None:
            dim1_path = str(self.scene_root / im1_path).replace('.jpg',
                                                                f'.{self.estimated_depth}.png')
            dim2_path = str(self.scene_root / im2_path).replace('.jpg',
                                                                f'.{self.estimated_depth}.png')
            depth1 = read_depth_image(dim1_path)
            depth2 = read_depth_image(dim2_path)
        else:
            depth1 = depth2 = torch.tensor([])

        # get absolute pose of im0 and im1
        # quaternion and translation vector that transforms World-to-Cam
        q1, t1 = self.poses[im1_path]
        # quaternion and translation vector that transforms World-to-Cam
        q2, t2 = self.poses[im2_path[-1]]  # the last frame is the query frame
        c1 = rotate_vector(-t1, qinverse(q1))  # center of camera 1 in world coordinates
        c2 = rotate_vector(-t2, qinverse(q2))  # center of camera 2 in world coordinates

        # get 4 x 4 relative pose transformation matrix (from im1 to im2)
        # for test/val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
        q12 = qmult(q2, qinverse(q1))
        t12 = t2 - rotate_vector(t1, q12)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)

        data = {
            'image0': image1,  # (3, h, w)
            'depth0': depth1,  # (h, w)
            'image1': image2,  # (9, 3, h, w)
            'depth1': depth2,  # (9, h, w)
            'T_0to1': T,  # (4, 4)  # relative pose
            'abs_q_0': q1,  # w2c
            'abs_c_0': c1,  # c2w
            'abs_q_1': q2,  # w2c
            'abs_c_1': c2,  # c2w
            'K_color0': self.K[im1_path].copy(),  # (3, 3)
            'K_color1': self.K[im2_path[-1]].copy(),  # (3, 3)
            'dataset_name': 'Mapfree',
            'scene_id': self.scene_root.stem,
            'scene_root': str(self.scene_root),
            'pair_id': index*self.sample_factor,
            'pair_names': (im1_path, im2_path),
            'sim': 0.  # needed for 7Scenes eval compatibility
        }

        if self.poses_device is not None:
            q1_device, t1_device = zip(*(self.poses_device[im2_path_] for im2_path_ in im2_path))
            data['abs_q_1_w2c_device'] = torch.from_numpy(np.stack(q1_device))
            data['abs_q_1_c2w_device'] = torch.from_numpy(
                np.stack([qinverse(q1_device_) for q1_device_ in q1_device]))
            data['abs_c_1_c2w_device'] = torch.from_numpy(np.stack(t1_device))
            q1_multi, t1_multi = zip(*(self.poses_device[im2_path_] for im2_path_ in im2_path))
            q1_multi_c2w = [qinverse(q1_multi_) for q1_multi_ in q1_multi]
            t1_multi_c2w = [rotate_vector(-t1_multi_, q1_multi_)
                            for q1_multi_, t1_multi_ in zip(q1_multi_c2w, t1_multi)]
            data['abs_q_1_c2w_multi'] = torch.from_numpy(np.stack(q1_multi_c2w))
            data['abs_c_1_c2w_multi'] = torch.from_numpy(np.stack(t1_multi_c2w))

        return data


class MapFreeDataset(data.ConcatDataset):
    def __init__(self, cfg, mode, transforms=None):
        assert mode in ['train', 'val', 'test'], 'Invalid dataset mode'

        scenes = cfg.DATASET.SCENES
        data_root = Path(cfg.DATASET.DATA_ROOT) / mode
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        # If None, no depth. Otherwise, loads depth map with name `frame_00000.suffix.png` where suffix is estimated_depth
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH
        overlap_limits = (cfg.DATASET.MIN_OVERLAP_SCORE, cfg.DATASET.MAX_OVERLAP_SCORE)
        assert isinstance(cfg.DATASET.QUERY_FRAME_COUNT, int)

        if cfg.DATASET.QUERY_FRAME_COUNT == 1:
            sample_factor = {'train': 1, 'val': 5, 'test': 5}[mode]
            sample_offset = 0
            SceneClass = MapFreeScene
        else:
            sample_factor = cfg.DATASET.QUERY_FRAME_COUNT + 1
            sample_offset = cfg.DATASET.QUERY_FRAME_COUNT  # the first frame to evaluate
            # e.g. from  1,  2,  3,  4,  5,  6,  7,  8,  9 predict the relative pose of 9, then
            #      from 11, 12, 13, 14, 15, 16, 17, 18, 19 predict the relative pose of 19
            if 9 != cfg.DATASET.QUERY_FRAME_COUNT:
                logger.warning('[WARNING] Query frame count is not 9, undefined behaviour!')
            SceneClass = MapFreeSceneMultiFrame

        if scenes is None:
            # Locate all scenes of the current dataset
            scenes = [s.name for s in data_root.iterdir() if s.is_dir()]
        else:
            scenes = [s for s in scenes if (data_root / s).exists()]

        # Init dataset objects for each scene
        data_srcs = [
            SceneClass(
                scene_root=data_root / scene, resize=resize, sample_factor=sample_factor,
                overlap_limits=overlap_limits, transforms=transforms,
                estimated_depth=estimated_depth, sample_offset=sample_offset) for scene in scenes]
        super().__init__(data_srcs)
