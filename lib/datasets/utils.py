import cv2
import numpy as np
import torch
from numpy.linalg import inv


def imread(path, augment_fn=None):
    cv_type = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), cv_type)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w, 3)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def read_color_image(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (3, h, w)
    """
    # read and resize image
    image = imread(path, None)
    image = cv2.resize(image, resize)

    # (h, w, 3) -> (3, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
    if augment_fn:
        image = augment_fn(image)
    return image


def read_depth_image(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.

    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path, color=True):
    """
    Read ScanNet's intrinsic matrix and returns 3x3 matrix. If color is True, returns color camera intrinsics.
    Otherwise returns depth camera intrinsics.
    The file containing the intrinsics is located in {scannet_root}/scans/scene{id}/sensor_data/_info.txt
    This file has the intrinsics of the depth camera and color camera under the keys 'm_calibrationColorIntrinsic'
    and 'm_calibrationDepthIntrinsic'.
    """

    key = 'm_calibrationColorIntrinsic' if color else 'm_calibrationDepthIntrinsic'

    with open(path, 'r') as f:
        for line in f.readlines():
            if key in line:
                mat = line.split(' = ')[1]
                mat = mat.lstrip().rstrip().split(' ')
                mat = [float(m) for m in mat]
                return np.array(mat).reshape(4, 4)[:-1, :-1]

    raise Exception(f'Invalid key {key}')


def correct_intrinsic_scale(K, scale_x, scale_y):
    '''Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
    the new coordinates x' = scale_x * x; y' = scale_y * y
    Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    '''

    transform = np.eye(3)
    transform[0, 0] = scale_x
    transform[0, 2] = scale_x / 2 - 0.5
    transform[1, 1] = scale_y
    transform[1, 2] = scale_y / 2 - 0.5
    Kprime = transform @ K

    return Kprime
