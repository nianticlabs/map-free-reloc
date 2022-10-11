import torch
import numpy as np
from collections import defaultdict


def pose_error_torch(R, t, Tgt, reduce=None):
    """Compute angular, scale and euclidean error of translation vector (metric). Compute angular rotation error."""

    Rgt = Tgt[:, :3, :3]                  # [B, 3, 3]
    tgt = Tgt[:, :3, 3:].transpose(1, 2)  # [B, 1, 3]

    scale_t = torch.linalg.norm(t, dim=-1)
    scale_tgt = torch.linalg.norm(tgt, dim=-1)

    cosine = (t @ tgt.transpose(1, 2)).squeeze(-1) / (scale_t * scale_tgt + 1e-9)
    cosine = torch.clip(cosine, -1.0, 1.0)    # handle numerical errors
    t_ang_err = torch.rad2deg(torch.acos(cosine))
    t_ang_err = torch.minimum(t_ang_err, 180 - t_ang_err)

    t_scale_err = scale_t / scale_tgt
    t_scale_err_sym = torch.maximum(scale_t / scale_tgt, scale_tgt / scale_t)
    t_euclidean_err = torch.linalg.norm(t - tgt, dim=-1)

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -1., 1.)  # handle numerical errors
    R_err = torch.rad2deg(torch.acos(cosine))

    if reduce is None:
        def fn(x): return x
    elif reduce == 'mean':
        fn = torch.mean
    elif reduce == 'median':
        fn = torch.median

    t_ang_err = fn(t_ang_err)
    t_scale_err = fn(t_scale_err)
    t_euclidean_err = fn(t_euclidean_err)
    R_err = fn(R_err)

    errors = {'t_err_ang': t_ang_err,
              't_err_scale': t_scale_err,
              't_err_scale_sym': t_scale_err_sym,
              't_err_euc': t_euclidean_err,
              'R_err': R_err}
    return errors


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = np.nan_to_num(errors, nan=float('inf'))   # convert nans to inf
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def ecdf(x):
    """Get Empirical Cumulative Distribution Function (ECDF) given samples x [N,]"""
    cd = np.linspace(0, 1, x.shape[0])
    v = np.sort(x)
    return v, cd


def print_auc_table(agg_metrics):
    pose_error = np.maximum(agg_metrics['R_err'], agg_metrics['t_err_ang'])
    auc_pose = error_auc(pose_error, (5, 10, 20))
    print('Pose error AUC @ 5/10/20deg: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_pose.values()))

    auc_rotation = error_auc(agg_metrics['R_err'], (5, 10, 20))
    print('Rotation error AUC @ 5/10/20deg: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_rotation.values()))

    auc_translation_ang = error_auc(agg_metrics['t_err_ang'], (5, 10, 20))
    print(
        'Translation angular error AUC @ 5/10/20deg: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_translation_ang.values()))

    auc_translation_euc = error_auc(agg_metrics['t_err_euc'], (0.1, 0.5, 1))
    print(
        'Translation Euclidean error AUC @ 0.1/0.5/1m: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_translation_euc.values()))


def precision(agg_metrics, rot_threshold, trans_threshold):
    '''Provides ratio of samples with rotation error < rot_threshold AND translation error < trans_threshold'''
    mask_rot = agg_metrics['R_err'] <= rot_threshold
    mask_trans = agg_metrics['t_err_euc'] <= trans_threshold
    recall = (mask_rot * mask_trans).mean()
    return recall


def A_metrics(t_scale_err_sym):
    """Returns A1/A2/A3 metrics of translation vector norm given the "symmetric" scale error
    where
    t_scale_err_sym = torch.maximum((t_norm_gt / t_norm_pred), (t_norm_pred / t_norm_gt))
    """

    if not torch.is_tensor(t_scale_err_sym):
        t_scale_err_sym = torch.from_numpy(t_scale_err_sym)

    thresh = t_scale_err_sym
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    return a1, a2, a3


class MetricsAccumulator:
    """Accumulates metrics and aggregates them when requested"""

    def __init__(self):
        self.data = defaultdict(list)

    def accumulate(self, data):
        for key, value in data.items():
            self.data[key].append(value)

    def aggregate(self):
        res = dict()
        for key in self.data.keys():
            res[key] = torch.cat(self.data[key]).view(-1).cpu().numpy()
        return res
