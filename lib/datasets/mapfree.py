import concurrent.futures
import re
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from pytorch_lightning import _logger as logger
from tqdm import tqdm
from transforms3d.quaternions import qinverse, qmult, quat2mat, rotate_vector

from lib.datasets.utils import correct_intrinsic_scale, read_color_image, read_depth_image
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

            if 0 < sample_offset:  # multi-frame case, needs to happen before masking
                valid_frame_ids = {
                    0: sorted(
                        # pairs from the first two columns with the pattern (0, imgId)
                        set(idxs[idxs[:, 0] == 0, 1])
                        # pairs from the second two columns with the pattern (0, imgId)
                        | set(idxs[idxs[:, 2] == 0, 3])),
                    1: sorted(
                        # pairs from the first two columns with the pattern (1, imgId)
                        set(idxs[idxs[:, 0] == 1, 1])
                        # pairs from the second two columns with the pattern (1, imgId)
                        | set(idxs[idxs[:, 2] == 1, 3]))
                }
                # reverse lookup: imgId -> linear ID in valid_frame_ids
                img_id_to_valid_frame_ids = {
                    0: {imgA: i for i, imgA in enumerate(valid_frame_ids[0])},
                    1: {imgB: i for i, imgB in enumerate(valid_frame_ids[1])}}

            if overlap_limits is not None:
                min_overlap, max_overlap = overlap_limits
                mask = np.logical_and((min_overlap < overlaps), (overlaps < max_overlap))
                idxs = idxs[mask]

            if 0 == sample_offset:  # single frame case
                assert sample_factor == 1
            else:  # multi frame case
                idxs_multi = [
                    (  # a row in idxs is a tuple of 4
                        seqA, imgA, seqB,
                        tuple(
                            valid_frame_ids_B[idx_in_valid_frame_ids - sample_offset + 1 + i]
                            for i in range(sample_offset)
                        ),
                    )  # end of row in idxs
                    for seqA, imgA, seqB, imgB in idxs
                    if (
                        # cache dict lookup
                        ((valid_frame_ids_B := valid_frame_ids[seqB]) is not None)
                        # the previous 8 frames start earliest from 0
                        and (0 <= (idx_in_valid_frame_ids := img_id_to_valid_frame_ids[seqB][imgB])
                                  - sample_offset + 1)
                        and (
                            # and either they are from different sequences
                            (seqA != seqB)
                            # or the map imgA frame does **not** fall into the span [imgB-8...imgB]
                            or (imgA
                                < valid_frame_ids_B[idx_in_valid_frame_ids - sample_offset + 1])
                            or (imgB < imgA)
                        )  # end of and
                    )  # end of if
                ]  # end of list comprehension

                idxs = idxs_multi
                del idxs_multi

            # TODO: figure out why copy is needed
            return idxs.copy()
        else:  # val and test case
            idxs = np.zeros((len(self.poses) - 1, 4), dtype=np.uint16)
            idxs[:, 2] = 1
            # match number between '_' and '.' in the filename, e.g. 'seq1/frame_00001.jpg'
            pattern = r"_(\d+)\..*$"
            idxs[:, 3] = np.array(
                sorted((  # sorted is not strictly needed
                    re.search(pattern, fn).group(1)
                    for fn in self.poses.keys()
                    if "seq0" not in fn
                )),
                dtype=np.uint16,
            )

            if 0 == sample_offset:  # single frame case
                # just filter
                idxs = idxs[sample_offset::sample_factor]
                assert sample_factor == 5
            else:  # multi frame case
                # remember chosen linear IDs in idxs
                # example: [9, 19, 29, ...]
                idxs_indices = np.arange(len(idxs))[sample_offset::sample_factor]

                # construct reverse lookup imgId --> linear ID in idxs
                # example (s00460): {9: 9, 19: 19, 29: 29, 39: 39, ...}
                # example (s00470): {11: 9, 21: 19, 31: 29, 41: 39, ...}
                imgB_to_idxs_indices = {
                    idxs[idxs_index, 3]: idxs_index for idxs_index in idxs_indices
                }

                # perform filtering and sequencing in one step
                # example (s00460): [(0, 0, 1, (1, 2, 3, 4, 5, 6, 7, 8, 9)),
                #                    (0, 0, 1, (11, 12, 13, 14, 15, 16, 17, 18, 19)),
                #                    ...]
                # example (s00470): [(0, 0, 1, (1, 3, 5, 6, 7, 8, 9, 10, 11)),
                #                    (0, 0, 1, (13, 14, 15, 16, 17, 18, 19, 20, 21)),
                #                    ...]
                idxs = [
                    (
                        seqA,
                        imgA,
                        seqB,
                        tuple(
                            idxs[i, 3]  # get imgB from the fourth column
                            for i in range(
                                idxs_index - sample_offset + 1,  # 9 - 9 + 1 = 1
                                idxs_index + 1,  # 9 + 1 = 10
                            )  # get rows 1 to 10-1 from the unfiltered idxs
                        ),
                    )
                    for seqA, imgA, seqB, imgB in idxs[sample_offset::sample_factor]
                    # this should never happen, but allows us to do the dict lookup only once
                    if sample_offset <= (idxs_index := imgB_to_idxs_indices[imgB])
                ]
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
        self.poses_device = self.read_poses(scene_root=self.scene_root, filename='poses_device.txt')

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
            if 'multi' in cfg.MODEL.lower():
                raise ValueError(f"Model {cfg.MODEL} is not compatible with a single frame dataset!")
            sample_factor = {'train': 1, 'val': 5, 'test': 5}[mode]
            sample_offset = 0
            SceneClass = MapFreeScene
        else:
            if 'multi' not in cfg.MODEL.lower():
                raise ValueError(f"Model {cfg.MODEL} is not compatible with a multi frame dataset!")
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.TRAINING.NUM_WORKERS) \
                as executor:
            futures = [executor.submit(SceneClass,
                                       scene_root=data_root / scene,
                                       resize=resize,
                                       sample_factor=sample_factor,
                                       overlap_limits=overlap_limits,
                                       transforms=transforms,
                                       estimated_depth=estimated_depth,
                                       sample_offset=sample_offset)
                       for scene in scenes]
            data_srcs = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"Loading {mode} scenes"):
                data_srcs.append(future.result())

        super().__init__(data_srcs)
