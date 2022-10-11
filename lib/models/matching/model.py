import torch

from lib.models.matching.feature_matching import *
from lib.models.matching.pose_solver import *


class FeatureMatchingModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.FEATURE_MATCHING == 'SIFT':
            self.feature_matching = SIFTMatching(cfg)
        elif cfg.FEATURE_MATCHING == 'Precomputed':
            self.feature_matching = PrecomputedMatching(cfg)
        else:
            raise NotImplementedError('Invalid feature matching')

        if cfg.POSE_SOLVER == 'EssentialMatrix':
            self.pose_solver = EssentialMatrixSolver(cfg)
        elif cfg.POSE_SOLVER == 'EssentialMatrixMetric':
            self.pose_solver = EssentialMatrixMetricSolver(cfg)
        elif cfg.POSE_SOLVER == 'Procrustes':
            self.pose_solver = ProcrustesSolver(cfg)
        elif cfg.POSE_SOLVER == 'PNP':
            self.pose_solver = PnPSolver(cfg)
        else:
            raise NotImplementedError('Invalid pose solver')

    def forward(self, data):
        assert data['depth0'].shape[0] == 1, 'Baseline models require batch size of 1'

        # get 2D-2D correspondences
        pts1, pts2 = self.feature_matching.get_correspondences(data)

        # get relative pose
        R, t, inliers = self.pose_solver.estimate_pose(pts1, pts2, data)
        data['inliers'] = inliers
        R = torch.from_numpy(R.copy()).unsqueeze(0).float()
        t = torch.from_numpy(t.copy()).view(1, 3).unsqueeze(0).float()
        return R, t
