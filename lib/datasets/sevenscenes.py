# Based on https://github.com/GrumpyZhou/visloc-relapose/blob/master/utils/datasets/relapose.py

import os
import glob

import torch
import torch.utils.data as data
import numpy as np
from scipy.spatial.transform import Rotation

from lib.datasets.utils import read_color_image, read_depth_image, correct_intrinsic_scale


class SceneDataset(data.Dataset):
    def __init__(self, scene_root, pair_txt, resize, transforms=None, one_nn=False,
                 estimated_depth=None):
        ''' scene_root: path to scene folder
            pair_txt: path to file specifying the (reference,query) pairs
            resize: shape to resize images
            transforms: function to apply to images
            one_nn: if True, keep only the reference image with highest DVLAD similarity to each query
        '''
        self.scene_root = scene_root
        self.transforms = transforms
        self.resize = resize
        self.estimated_depth = estimated_depth

        # load relative poses for given pairs
        self.im_pairs, self.relv_poses, _, self.sim = self.parse_relv_pose_txt(os.path.join(
            scene_root,
            pair_txt))
        self.original_idxs = list(range(len(self.im_pairs)))
        if one_nn:
            self.filter_one_nn()
        self.num = len(self.im_pairs)

        # load absolute poses for each sample
        self.abs_poses = self.parse_abs_pose_txt(os.path.join(scene_root, 'dataset_test.txt'))
        self.abs_poses.update(self.parse_abs_pose_txt(
            os.path.join(scene_root, 'dataset_train.txt')))

        # static intrinsic matrix
        ox, oy = 320, 240
        f = 525
        self.K = np.array([[f, 0, ox], [0, f, oy], [0, 0, 1]], dtype=np.float32)
        self.K = correct_intrinsic_scale(self.K, resize[0] / 640, resize[1] / 480)

    def parse_relv_pose_txt(self, fpath, with_ess=False):
        '''Relative pose pair format:image1 image2 sim w p q r x y z ess_vec'''
        im_pairs = []
        ess_vecs = [] if with_ess else None
        relv_poses = []
        sim = []
        with open(fpath) as f:
            for line in f:
                cur = line.split()
                im_pairs.append((cur[0], cur[1]))
                sim.append(float(cur[2]))
                q = np.array([float(i) for i in cur[3:7]], dtype=np.float32)
                t = np.array([float(i) for i in cur[7:10]], dtype=np.float32)

                # change q convention to [x, y, z, w]
                q = q[[1, 2, 3, 0]]
                R = Rotation.from_quat(q).as_matrix()

                # Convert to rotation matrix and 4x4 pose matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, -1] = t.ravel()
                relv_poses.append(T)

                if with_ess:
                    ess_vecs.append(np.array([float(i) for i in cur[10:19]], dtype=np.float32))
        return im_pairs, relv_poses, ess_vecs, sim

    def parse_abs_pose_txt(self, fpath):
        """Absolute pose label format:
            3 header lines
            list of samples with format:
                image x y z w p q r
        """

        pose_dict = {}
        with open(fpath) as f:
            for line in f.readlines()[3::]:  # Skip 3 header lines
                cur = line.split(' ')
                c = np.array([float(v) for v in cur[1:4]], dtype=np.float32)
                q = np.array([float(v) for v in cur[4:8]], dtype=np.float32)
                im = cur[0]
                pose_dict[im] = (c, q)
        return pose_dict

    def filter_one_nn(self):
        """Filters pairs such that for each query image, only the reference image with highest similarity is kept"""

        kept_queries_idx = {}  # dict (query image, kept_idx)
        kept_queries_sim = {}  # dict (query image, kept_similarity)

        for i, ((ref, query), sim) in enumerate(zip(self.im_pairs, self.sim)):
            if query in kept_queries_sim:
                if sim < kept_queries_sim[query]:
                    continue

            kept_queries_idx[query] = i
            kept_queries_sim[query] = sim

        # update internal arrays
        keep_idxs = list(kept_queries_idx.values())
        self.im_pairs = [self.im_pairs[idx] for idx in keep_idxs]
        self.relv_poses = [self.relv_poses[idx] for idx in keep_idxs]
        self.sim = [self.sim[idx] for idx in keep_idxs]
        self.original_idxs = keep_idxs

    def __getitem__(self, index):
        # load color images
        im1_path, im2_path = [os.path.join(self.scene_root, im_ref)
                              for im_ref in self.im_pairs[index]]
        image1 = read_color_image(im1_path, self.resize, augment_fn=self.transforms)
        image2 = read_color_image(im2_path, self.resize, augment_fn=self.transforms)

        # load depth maps
        depth_path_suffix = '.depth.' if self.estimated_depth is None else f'.depth.{self.estimated_depth}.'
        dim1_path = im1_path.replace('.color.', depth_path_suffix)
        dim2_path = im2_path.replace('.color.', depth_path_suffix)
        depth1 = read_depth_image(dim1_path)
        depth2 = read_depth_image(dim2_path)

        # get relative pose transformation
        T_0to1 = torch.tensor(self.relv_poses[index], dtype=torch.float32)

        # get absolute pose of im0 and im1
        im1ref, im2ref = self.im_pairs[index]
        # center of camera 1 in world coordinates, quaternion transf. from camera to world
        c1, q1 = self.abs_poses[im1ref]
        # center of camera 2 in world coordinates, quaternion transf. from camera to world
        c2, q2 = self.abs_poses[im2ref]

        data = {
            'image0': image1,  # (3, h, w)
            'depth0': depth1,  # (h, w)
            'image1': image2,
            'depth1': depth2,
            'T_0to1': T_0to1,  # (4, 4)  # relative pose
            'abs_q_0': q1,
            'abs_c_0': c1,
            'abs_q_1': q2,
            'abs_c_1': c2,
            'sim': self.sim[index],  # DVLAD similarity
            'K_color0': self.K.copy(),  # (3, 3)
            'K_color1': self.K.copy(),  # (3, 3)
            'K_depth': self.K.copy(),  # (3, 3)
            'dataset_name': '7Scenes',
            'scene_id': self.scene_root.split('/')[-1],
            'scene_root': str(self.scene_root),
            'pair_id': self.original_idxs[index],
            'pair_names': self.im_pairs[index]
        }

        return data

    def __len__(self):
        return self.num


class SevenScenesDataset(data.ConcatDataset):
    def __init__(self, cfg, mode, transforms=None):

        scenes = cfg.DATASET.SCENES
        data_root = cfg.DATASET.DATA_ROOT
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        # If None, loads GT depth. Otherwise, loads depth map with name `pairs.depth.suffix.png` where suffix is estimated_depth
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH

        assert mode in ['train', 'val', 'test'], 'Invalid dataset mode'
        pair_txt = {'train': cfg.DATASET.PAIRS_TXT.TRAIN,
                    'val': cfg.DATASET.PAIRS_TXT.VAL,
                    'test': cfg.DATASET.PAIRS_TXT.TEST}[mode]
        one_nn = cfg.DATASET.PAIRS_TXT.ONE_NN

        if scenes is None:
            # Locate all scenes of the current dataset
            scenes = self.glob_scenes(data_root, pair_txt)

        # Init dataset objects for each scene
        data_srcs = [
            SceneDataset(
                os.path.join(data_root, scene),
                pair_txt, resize, transforms, one_nn, estimated_depth) for scene in scenes]
        super().__init__(data_srcs)

    def glob_scenes(self, data_root, pair_txt):
        scenes = []
        for sdir in glob.iglob('{}/*/{}'.format(data_root, pair_txt)):
            sdir = sdir.split('/')[-2]
            scenes.append(sdir)
        return sorted(scenes)
