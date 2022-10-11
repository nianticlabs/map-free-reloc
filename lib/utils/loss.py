import math
import inspect

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from kornia.geometry.conversions import rotation_matrix_to_quaternion, QuaternionCoeffOrder


def data_wrapper(func):
    """Decorator that obtains the functions arguments from the shared 'data' dictionary
    Allows loss functions to be self-contained without direct references to 'data' 
    """
    def wrapped(data):
        # get arguments names
        arg_list = list(inspect.signature(func).parameters)

        # fill dict with arguments names and their values (from shared data dict)
        arguments = {'R': data['R'],
                     't': data['t'],
                     'Rgt': data['T_0to1'][:, :3, :3],
                     'tgt': data['T_0to1'][:, :3, 3:].transpose(1, 2)
                     }

        # add quaternion ground-truth, if using quat. loss functions
        if 'q' in arg_list:
            arguments['q'] = data['q']
            qgt = rotation_matrix_to_quaternion(
                arguments['Rgt'].contiguous(),
                order=QuaternionCoeffOrder.WXYZ)
            # enforces using a single quaternion hemishehere (avoiding q, -q duble representation)
            qgt *= torch.sign(qgt[:, 0:1])
            arguments['qgt'] = qgt

        # add scale, if using specific loss functions
        if 'scale' in arg_list:
            arguments['scale'] = data['scale']
            arguments['scalegt'] = torch.linalg.norm(arguments['tgt'], dim=-1).unsqueeze(-1)

        # add t_direction, if using specific loss functions
        if 't_direction' in arg_list:
            arguments['t_direction'] = data['t_direction']
            arguments['t_directiongt'] = F.normalize(arguments['tgt'], dim=-1)

        # R_bins from AngularBin head
        if 'R_bins' in arg_list:
            arguments['R_bins'] = data['R_bins']
            R_binsgt = torch.from_numpy(
                Rotation.from_matrix(arguments['Rgt'].cpu().numpy()).as_euler(
                    'xyz', degrees=True))  # [B, 3]
            # add offset to get interval [0, 360] in XZ and [0,180] in Y
            R_binsgt += torch.FloatTensor([[180, 90, 180]])
            R_binsgt = torch.round(R_binsgt).long()
            R_binsgt[:, 0] = torch.clamp(R_binsgt[:, 0], 0, 359)  # clamps to fit in bins
            R_binsgt[:, 1] = torch.clamp(R_binsgt[:, 1], 0, 179)
            R_binsgt[:, 2] = torch.clamp(R_binsgt[:, 2], 0, 359)
            arguments['R_binsgt'] = R_binsgt.to(arguments['Rgt'].device)

        # spherical angles of translation vector, from AngularBin head
        if 't_sph_phi' in arg_list or 't_sph_theta' in arg_list:
            arguments['t_sph_phi'] = data['t_sph_phi']
            arguments['t_sph_theta'] = data['t_sph_theta']

            t_direction_gt = F.normalize(arguments['tgt'], dim=-1).reshape(-1, 3)
            t_sph_theta_gt = torch.acos(t_direction_gt[:, 2])
            t_sph_phi_gt = torch.atan2(t_direction_gt[:, 1], t_direction_gt[:, 0] + 1e-5)
            t_sph_phi_gt[t_sph_phi_gt < 0] += 2*math.pi
            t_sph_theta_gt = torch.clamp(torch.round(torch.rad2deg(t_sph_theta_gt)).long(), 0, 179)
            t_sph_phi_gt = torch.round(torch.rad2deg(t_sph_phi_gt)).long()
            t_sph_phi_gt[t_sph_phi_gt == 360] = 0
            arguments['t_sph_phigt'] = t_sph_phi_gt
            arguments['t_sph_thetagt'] = t_sph_theta_gt

        # get argument values and returns function result on arguments
        arg_value = [arguments[x] for x in arg_list]
        return func(*arg_value)
    return wrapped


@data_wrapper
def rot_frobenius_loss(R, Rgt):
    """Computes rotation loss using Frobenius norm.
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    B = R.shape[0]
    eye_batch = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(R.device)
    R_residual = Rgt.transpose(1, 2) @ R
    R_loss = F.mse_loss(R_residual, eye_batch)
    return R_loss


@data_wrapper
def rot_l1_loss(R, Rgt):
    """Computes rotation loss using L1 norm over residual rotation matrix.
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    B = R.shape[0]
    eye_batch = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(R.device)
    R_residual = Rgt.transpose(1, 2) @ R
    R_loss = F.l1_loss(R_residual, eye_batch)
    return R_loss


@data_wrapper
def rot_angle_loss(R, Rgt):
    """
    Computes rotation loss using L2 error of residual rotation angle [radians]
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    R_err = torch.acos(cosine)
    loss = F.l1_loss(R_err, torch.zeros_like(R_err))
    return loss


@data_wrapper
def rot_bin_loss(R_bins, R_binsgt):
    lrx = F.cross_entropy(R_bins[:, :360], R_binsgt[:, 0])
    lry = F.cross_entropy(R_bins[:, 360:540], R_binsgt[:, 1])
    lrz = F.cross_entropy(R_bins[:, 540:], R_binsgt[:, 2])
    return (lrx + lry + lrz) / 3


@data_wrapper
def trans_l2_loss(t, tgt):
    """Computes L2 loss for translation vector
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    return F.mse_loss(t, tgt)


@data_wrapper
def trans_l1_loss(t, tgt):
    """Computes L1 loss for translation vector
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    return F.l1_loss(t, tgt)


@data_wrapper
def quat_l1_loss(q, qgt):
    """Computes L1 loss between quaternions
    Input:
    q - estimated quaternion [B, 4]
    qgt - ground-truth quaternion [B, 4]
    Output: quat. loss
    """
    return F.l1_loss(q, qgt)


@data_wrapper
def robust_quat_l1_loss(q, qgt):
    """Robust L1 quaternion loss.

    q - estimated quaternion [B, 4]
    qgt - ground-truth quaternion [B, 4]

    Source: https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf
    page 10: "Quaternion distance"

    Note: probably assumes normalized quaternion, which for us is true for targ.

    Note2: min(||pred - targ||_2, ||pred + targ||_2)^2 would be a *non-robust* L2 loss.
    """
    assert q.shape[1] == 4
    assert qgt.shape[1] == 4
    return torch.mean(
        torch.minimum(torch.linalg.norm(q + qgt, dim=1, keepdim=True),
                      torch.linalg.norm(q - qgt, dim=1, keepdim=True)))


@data_wrapper
def trans_scale_direction_loss(scale, scalegt, t_direction, t_directiongt):
    """ Computes translation loss in two componentes: scale loss (L1) and t_direction loss (angular loss)
    Input:
    scale - estimated scale [B, 1, 1]
    scalegt - ground-truth scale [B, 1, 1]
    t_direction - estimated translation direction (unitary) [B, 1, 3]
    t_directiongt - ground-truth translation direction (unitary) [B, 1, 3]
    """
    return F.l1_loss(scale, scalegt) + F.l1_loss(t_direction, t_directiongt)


@data_wrapper
def trans_ang_loss(t, tgt):
    """Computes L1 loss for translation vector ANGULAR error
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    scale_t = torch.linalg.norm(t, dim=-1)
    scale_tgt = torch.linalg.norm(tgt, dim=-1)

    cosine = (t @ tgt.transpose(1, 2)).squeeze(-1) / (scale_t * scale_tgt + 1e-6)
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    t_ang_err = torch.acos(cosine)
    t_ang_err = torch.minimum(t_ang_err, math.pi - t_ang_err)
    return F.l1_loss(t_ang_err, torch.zeros_like(t_ang_err))


@data_wrapper
def trans_sphbin_loss(t_sph_phi, t_sph_phigt, t_sph_theta, t_sph_thetagt, scale, scalegt):
    lscale = F.l1_loss(scale, scalegt)
    lphi = F.cross_entropy(t_sph_phi, t_sph_phigt)
    ltheta = F.cross_entropy(t_sph_theta, t_sph_thetagt)
    return lscale + (lphi + ltheta) / 2


@data_wrapper
def trans_scale_l1_loss(scale, scalegt):
    return F.l1_loss(scale, scalegt)


@data_wrapper
def empty_loss(tgt):
    return torch.zeros(1, device=tgt.device, dtype=torch.float32)
