import torch
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder
from scipy.spatial.transform import Rotation

from lib.models.regression.encoder.preact import PreActBlock
from lib.utils.solver import procrustes
from lib.utils.rotationutils import rotation_matrix_from_ortho6d


class ResBlockMLP(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        H, W = cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH
        self.resblock1 = PreActBlock(in_channels, 256, stride=2)
        self.resblock2 = PreActBlock(256, 128, stride=2)

    def forward(self, feature_volume):
        B = feature_volume.shape[0]

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = x.view(B, -1)
        return x


class DeepResBlock(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        bn = cfg.HEAD.BATCH_NORM
        self.avg_pool = cfg.HEAD.AVG_POOL
        self.resblock1 = PreActBlock(in_channels, 64, stride=2, bn=bn)
        self.resblock2 = PreActBlock(64, 128, stride=2, bn=bn)
        self.resblock3 = PreActBlock(128, 256, stride=2, bn=bn)
        self.resblock4 = PreActBlock(256, 512, stride=2, bn=bn)

    def forward(self, feature_volume):
        B = feature_volume.shape[0]

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        if self.avg_pool:
            x = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        x = x.view(B, -1)
        return x


class ProcrustesResBlockMLP(ResBlockMLP):
    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.add_basis = cfg.HEAD.ADD_BASIS
        self.num_pts = cfg.HEAD.NUM_PTS
        assert self.num_pts == 3 or (self.num_pts % 2 == 0 and self.num_pts >=
                                     6), 'num_pts must be 3, 6 or a multiple of 2 higher than 6'

        self.mlp = torch.nn.LazyLinear(3 * self.num_pts, bias=True)

    def forward(self, feature_volume, data):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        xyz = self.mlp(x).view(B, -1, 3)

        basis = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(x.device)

        # get correspondences
        if self.num_pts == 3:
            cor0 = basis
            cor1 = xyz[:, :]
        else:
            cor0 = xyz[:, :self.num_pts//2]
            cor1 = xyz[:, self.num_pts//2:]

        if self.add_basis:
            cor0 = cor0 + basis if self.num_pts == 6 else cor0
            cor1 = cor1 + basis if self.num_pts in (3, 6) else cor1

        # get relative pose
        R, t = procrustes(cor0, cor1)

        # check validity
        self.xyz = xyz.detach().clone()
        self.R = R.detach().clone()
        self.t = t.detach().clone()
        invalid_anchors = (torch.isnan(xyz).any() or torch.isinf(xyz).any())
        invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
        invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())
        if invalid_anchors:
            print('Invalid anchors!')
        if invalid_R:
            print('Invalid anchors!')
        if invalid_t:
            print('Invalid anchors!')
        if invalid_anchors or invalid_R or invalid_t:
            import sys
            sys.exit("Stopped")

        return R, t


class ProcrustesDeepResBlock(DeepResBlock):
    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.add_basis = cfg.HEAD.ADD_BASIS
        self.num_pts = cfg.HEAD.NUM_PTS
        assert self.num_pts == 3 or (self.num_pts % 2 == 0 and self.num_pts >=
                                     6), 'num_pts must be 3, 6 or a multiple of 2 higher than 6'

        self.mlp = torch.nn.Sequential(
            *[
                torch.nn.LazyLinear(256, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3 * self.num_pts, bias=True)
            ])

    def forward(self, feature_volume, data):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        xyz = self.mlp(x).view(B, -1, 3)

        basis = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(x.device)

        # get correspondences
        if self.num_pts == 3:
            cor0 = basis
            cor1 = xyz[:, :]
        else:
            cor0 = xyz[:, :self.num_pts//2]
            cor1 = xyz[:, self.num_pts//2:]

        if self.add_basis:
            cor0 = cor0 + basis if self.num_pts == 6 else cor0
            cor1 = cor1 + basis if self.num_pts in (3, 6) else cor1

        # get relative pose
        R, t = procrustes(cor0, cor1)

        # check validity
        self.xyz = xyz.detach().clone()
        self.R = R.detach().clone()
        self.t = t.detach().clone()
        invalid_anchors = (torch.isnan(xyz).any() or torch.isinf(xyz).any())
        invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
        invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())
        if invalid_anchors:
            print('Invalid anchors!')
        if invalid_R:
            print('Invalid anchors!')
        if invalid_t:
            print('Invalid anchors!')
        if invalid_anchors or invalid_R or invalid_t:
            import sys
            sys.exit("Stopped")

        return R, t


class QuatDeepResBlock(DeepResBlock):
    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        # if true, regress unitary translation vector (3D) + scale as a single scalar
        # else, regress scaled translation vector (3D)
        self.regress_scale = cfg.HEAD.SEPARATE_SCALE
        self.output_dims = 8 if self.regress_scale else 7

        self.mlp = torch.nn.Sequential(
            *[
                torch.nn.LazyLinear(256, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.output_dims, bias=True)
            ])

    def forward(self, feature_volume, data):
        """Regresses a quaternion (4D) + translation (unit 3D vector + 1D scale)"""
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        x = self.mlp(x).view(B, -1)

        quat = torch.nn.functional.normalize(x[:, :4], dim=1)
        data['q'] = quat
        R = quaternion_to_rotation_matrix(quat, order=QuaternionCoeffOrder.WXYZ)

        if self.regress_scale:
            scale = torch.abs(x[:, 4]).view(B, 1, 1)
            t_direction = torch.nn.functional.normalize(x[:, 5:], dim=1).view(B, 1, 3)
            t = scale * t_direction
            data['t_direction'] = t_direction
            data['scale'] = scale
        else:
            t = x[:, 4:].view(B, 1, 3)

        # check validity
        invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
        invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())
        if invalid_R:
            print('Invalid R!')
        if invalid_t:
            print('Invalid t!')
        if invalid_R or invalid_t:
            import sys
            sys.exit("Stopped")
        return R, t


class DirectResBlockMLP(ResBlockMLP):
    """
    Direct R,t estimation using continuous 6D rotation representation from
    On the Continuity of Rotation Representations in Neural Networks
    https://arxiv.org/pdf/1812.07035.pdf
    """

    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.mlp = torch.nn.LazyLinear(3 + 6, bias=True)

    def forward(self, feature_volume, data):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        out = self.mlp(x).view(B, 9)

        ortho = out[:, :6]
        R = rotation_matrix_from_ortho6d(ortho)
        t = out[:, 6:].view(B, 1, 3)
        return R, t


class DirectDeepResBlockMLP(DeepResBlock):
    """
    Direct R,t estimation using continuous 6D rotation representation from
    On the Continuity of Rotation Representations in Neural Networks
    https://arxiv.org/pdf/1812.07035.pdf
    """

    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.mlp = torch.nn.Sequential(
            *[
                torch.nn.LazyLinear(256, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3 + 6, bias=True)
            ])

    def forward(self, feature_volume, data):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        out = self.mlp(x).view(B, 9)

        ortho = out[:, :6]
        R = rotation_matrix_from_ortho6d(ortho)
        t = out[:, 6:].view(B, 1, 3)
        return R, t


class AngularBinsDeepResBlockMLP(DeepResBlock):
    """
    Trats R estimation as a classification problem: rotation is estimated by classifying each rotation angle (X,Y,Z)
     into a set of 360 bins (Euler angles X/Z) and 180 bins (Euler angles Y).
    Translation is estimated either by 1) direct regression 2) bins for spherical coordinates angles and regressing scale separately
     depending on argument HEAD.SEPARATE_SCALE
    """

    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.regress_scale_separately = cfg.HEAD.SEPARATE_SCALE

        output_dims = 360 * 2 + 180  # 3 rotation angles, 360 bins X/Z, 180 bins Y
        output_dims += 360 + 180 + 1 if self.regress_scale_separately else 3

        self.mlp = torch.nn.LazyLinear(output_dims, bias=True)

    def forward(self, feature_volume, data):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        out = self.mlp(x).view(B, -1)

        # convert rotation bins to Rotation mat.
        R_bins = out[:, :900].view(B, 900)
        data['R_bins'] = R_bins
        with torch.no_grad():
            angle_x = torch.argmax(R_bins[:, :360], dim=1, keepdim=True)
            angle_y = torch.argmax(R_bins[:, 360:540], dim=1, keepdim=True)
            angle_z = torch.argmax(R_bins[:, 540:], dim=1, keepdim=True)
            angles = torch.cat((angle_x, angle_y, angle_z), dim=1)  # [B, 3]
            # offset back to [-180,180]
            angles -= torch.LongTensor([[180, 90, 180]]).to(angles.device)
            R = Rotation.from_euler(
                'xyz', angles.cpu().numpy(),
                degrees=True).as_matrix()  # [B, 3, 3]
            R = torch.from_numpy(R).to(out.device).float()

        if self.regress_scale_separately:
            t_sph_phi = out[:, 900:1260]  # 360bins
            t_sph_theta = out[:, 1260:1440]  # 180 bins
            scale = torch.abs(out[:, -1:])  # 1D scale
            data['t_sph_phi'] = t_sph_phi
            data['t_sph_theta'] = t_sph_theta
            data['scale'] = scale.view(B, 1, 1)

            phi = torch.deg2rad(torch.argmax(t_sph_phi, dim=1).float()).view(B, -1)
            theta = torch.deg2rad(torch.argmax(t_sph_theta, dim=1).float()).view(B, -1)
            t = scale * torch.cat([torch.cos(phi)*torch.sin(theta), torch.sin(phi)
                                  * torch.sin(theta), torch.cos(theta)], dim=1)
        else:
            t = out[:, 900:]

        t = t.view(B, 1, 3)
        return R, t
