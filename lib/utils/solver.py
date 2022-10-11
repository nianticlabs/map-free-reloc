import torch


def procrustes(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (B, N, 3) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (B, N, 3) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation (B, 3, 3)
        -    t: optimal translation  (B, 3, 1)
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    assert len(A.shape) == len(B.shape) == 3, 'three dimensions are required'
    assert A.shape[0] == B.shape[0], 'batch size must match'
    assert A.shape[1] == B.shape[1], 'number of correspondences must match'
    assert A.shape[2] == B.shape[2], 'number of spatial dimensions must be 3'

    a_mean = A.mean(axis=1, keepdim=True)
    b_mean = B.mean(axis=1, keepdim=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.transpose(1, 2) @ B_c
    U, S, V = torch.svd(H)
    # Fixes orientation such that Det(R) = + 1
    Z = torch.eye(3).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    # Rotation matrix
    R = V @ Z @ U.transpose(1, 2)
    # Translation vector
    t = b_mean - a_mean @ R.transpose(1, 2)
    return R, t
