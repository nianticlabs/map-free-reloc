import os
import numpy as np


def load_scannet_imgpaths(npz_path, root_dir):
    data_names = np.load(npz_path)['name']
    pair_paths = []

    for scene_name, scene_sub_name, stem_name_0, stem_name_1 in data_names:
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        img_name0 = os.path.join(root_dir, scene_name, 'sensor_data',
                                 f'frame-{stem_name_0:06}.color.jpg')
        img_name1 = os.path.join(root_dir, scene_name, 'sensor_data',
                                 f'frame-{stem_name_1:06}.color.jpg')
        pair_paths.append((img_name0, img_name1))

    return pair_paths


def parse_7scenes_matching_pairs(pair_txt):
    """Get list of image pairs for matching
    Arg:
        pair_txt: file contains image pairs and essential
        matrix with line format
            image1 image2 sim w p q r x y z ess_vec
    Return:
        list of 3d-tuple contains (q=[wpqr], t=[xyz], essential matrix)
    """
    im_pairs = {}
    f = open(pair_txt)
    for line in f:
        cur = line.split()
        im1, im2 = cur[0], cur[1]
        q = np.array([float(i) for i in cur[3:7]], dtype=np.float32)
        t = np.array([float(i) for i in cur[7:10]], dtype=np.float32)
        ess_mat = np.array([float(i) for i in cur[10:19]], dtype=np.float32).reshape(3, 3)
        im_pairs[(im1, im2)] = (q, t, ess_mat)
    f.close()
    return im_pairs


def parse_mapfree_query_frames(pose_path):
    """
    Get list of query frames given a pose path
    :param pose_path:
    :return:
    """
    query_paths = []
    with pose_path.open('r') as f:
        for l in f.readlines():
            # skip if comment(#) or keyframe (seq0)
            if '#' in l or 'seq0' in l:
                continue
            qpath = l.strip().split(' ')[0]
            query_paths.append(qpath)
    return query_paths


def stack_pts(pts_list):
    '''Given a pts list with N arrays, each shaped (Npts, D), where Npts varies, creates a common array shaped (N, max(Npts), D) filled with NaNs when Npts < Max(Npts)'''
    assert len(pts_list) > 0, 'list must not be empty'

    N = len(pts_list)
    max_npts = max([pts.shape[0] for pts in pts_list])
    D = pts_list[0].shape[1]
    pts_stack = np.full((N, max_npts, D), np.nan)
    for i, pts in enumerate(pts_list):
        pts_stack[i, :pts.shape[0]] = pts
    return pts_stack
