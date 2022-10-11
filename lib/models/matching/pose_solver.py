import numpy as np
import cv2 as cv
import open3d as o3d


def backproject_3d(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    '''

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xyz = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
    return xyz


class EssentialMatrixSolver:
    '''Obtain relative pose (up to scale) given a set of 2D-2D correspondences'''

    def __init__(self, cfg):

        # EMat RANSAC parameters
        self.ransac_pix_threshold = cfg.EMAT_RANSAC.PIX_THRESHOLD
        self.ransac_confidence = cfg.EMAT_RANSAC.CONFIDENCE

    def estimate_pose(self, kpts0, kpts1, data):
        R = np.full((3, 3), np.nan)
        t = np.full((3, 1), np.nan)
        if len(kpts0) < 5:
            return R, t, 0

        K0 = data['K_color0'].squeeze(0).numpy()
        K1 = data['K_color1'].squeeze(0).numpy()

        # normalize keypoints
        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        # normalize ransac threshold
        ransac_thr = self.ransac_pix_threshold / np.mean([K0[0, 0], K1[1, 1], K0[1, 1], K1[0, 0]])

        # compute pose with OpenCV
        E, mask = cv.findEssentialMat(
            kpts0, kpts1, np.eye(3),
            threshold=ransac_thr, prob=self.ransac_confidence, method=cv.USAC_MAGSAC)
        self.mask = mask
        if E is None:
            return R, t, 0

        # recover pose from E
        best_num_inliers = 0
        ret = R, t, 0
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], n)
        return ret


class EssentialMatrixMetricSolverMEAN(EssentialMatrixSolver):
    '''Obtains relative pose with scale using E-Mat decomposition and depth values at inlier correspondences'''

    def __init__(self, cfg):
        super().__init__(cfg)

    def estimate_pose(self, kpts0, kpts1, data):
        '''Estimates metric translation vector using by back-projecting E-mat inliers to 3D using depthmaps.
        The metric translation vector can be obtained by looking at the residual vector (projected to the translation vector direction).
        In this version, each 3D-3D correspondence gives an optimal scale for the translation vector. 
        We simply aggregate them by averaging them.
        '''

        # get pose up to scale
        R, t, inliers = super().estimate_pose(kpts0, kpts1, data)
        if inliers == 0:
            return R, t, inliers

        # backproject E-mat inliers at each camera
        K0 = data['K_color0'].squeeze(0)
        K1 = data['K_color1'].squeeze(0)
        mask = self.mask.ravel() == 1        # get E-mat inlier mask from super class
        inliers_kpts0 = np.int32(kpts0[mask])
        inliers_kpts1 = np.int32(kpts1[mask])
        depth_inliers_0 = data['depth0'][0, inliers_kpts0[:, 1], inliers_kpts0[:, 0]].numpy()
        depth_inliers_1 = data['depth1'][0, inliers_kpts1[:, 1], inliers_kpts1[:, 0]].numpy()
        # check for valid depth
        valid = (depth_inliers_0 > 0) * (depth_inliers_1 > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        xyz0 = backproject_3d(inliers_kpts0[valid], depth_inliers_0[valid], K0)
        xyz1 = backproject_3d(inliers_kpts1[valid], depth_inliers_1[valid], K1)

        # rotate xyz0 to xyz1 CS (so that axes are parallel)
        xyz0 = (R @ xyz0.T).T

        # get average point for each camera
        pmean0 = np.mean(xyz0, axis=0)
        pmean1 = np.mean(xyz1, axis=0)

        # find scale as the 'length' of the translation vector that minimises the 3D distance between projected points from 0 and the corresponding points in 1
        scale = np.dot(pmean1 - pmean0, t)
        t_metric = scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, inliers


class EssentialMatrixMetricSolver(EssentialMatrixSolver):
    '''
        Obtains relative pose with scale using E-Mat decomposition and RANSAC for scale based on depth values at inlier correspondences.
        The scale of the translation vector is obtained using RANSAC over the possible scales recovered from 3D-3D correspondences.
    '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ransac_scale_threshold = cfg.EMAT_RANSAC.SCALE_THRESHOLD

    def estimate_pose(self, kpts0, kpts1, data):
        '''Estimates metric translation vector using by back-projecting E-mat inliers to 3D using depthmaps.
        '''

        # get pose up to scale
        R, t, inliers = super().estimate_pose(kpts0, kpts1, data)
        if inliers == 0:
            return R, t, inliers

        # backproject E-mat inliers at each camera
        K0 = data['K_color0'].squeeze(0)
        K1 = data['K_color1'].squeeze(0)
        mask = self.mask.ravel() == 1        # get E-mat inlier mask from super class
        inliers_kpts0 = np.int32(kpts0[mask])
        inliers_kpts1 = np.int32(kpts1[mask])
        depth_inliers_0 = data['depth0'][0, inliers_kpts0[:, 1], inliers_kpts0[:, 0]].numpy()
        depth_inliers_1 = data['depth1'][0, inliers_kpts1[:, 1], inliers_kpts1[:, 0]].numpy()

        # check for valid depth
        valid = (depth_inliers_0 > 0) * (depth_inliers_1 > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        xyz0 = backproject_3d(inliers_kpts0[valid], depth_inliers_0[valid], K0)
        xyz1 = backproject_3d(inliers_kpts1[valid], depth_inliers_1[valid], K1)

        # rotate xyz0 to xyz1 CS (so that axes are parallel)
        xyz0 = (R @ xyz0.T).T

        # get individual scales (for each 3D-3D correspondence)
        scale = np.dot(xyz1 - xyz0, t.reshape(3, 1))  # [N, 1]

        # RANSAC loop
        best_inliers = 0
        best_scale = None
        for scale_hyp in scale:
            inliers_hyp = (np.abs(scale - scale_hyp) < self.ransac_scale_threshold).sum().item()
            if inliers_hyp > best_inliers:
                best_scale = scale_hyp
                best_inliers = inliers_hyp

        # Output results
        t_metric = best_scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, best_inliers


class PnPSolver:
    '''Estimate relative pose (metric) using Perspective-n-Point algorithm (2D-3D) correspondences'''

    def __init__(self, cfg):
        # PnP RANSAC parameters
        self.ransac_iterations = cfg.PNP.RANSAC_ITER
        self.reprojection_inlier_threshold = cfg.PNP.REPROJECTION_INLIER_THRESHOLD
        self.confidence = cfg.PNP.CONFIDENCE

    def estimate_pose(self, pts0, pts1, data):
        # uses nearest neighbour
        pts0 = np.int32(pts0)

        if len(pts0) < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # get depth at correspondence points
        depth_0 = data['depth0'].squeeze(0)
        depth_pts0 = depth_0[pts0[:, 1], pts0[:, 0]]

        # remove invalid pts (depth == 0)
        valid = depth_pts0 > depth_0.min()
        if valid.sum() < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        pts0 = pts0[valid]
        pts1 = pts1[valid]
        depth_pts0 = depth_pts0[valid]

        # backproject points to 3D in each sensors' local coordinates
        K0 = data['K_color0'].squeeze(0)
        K1 = data['K_color1'].squeeze(0)
        xyz_0 = backproject_3d(pts0, depth_pts0, K0).numpy()

        # get relative pose using PnP + RANSAC
        succ, rvec, tvec, inliers = cv.solvePnPRansac(
            xyz_0, pts1, K1.numpy(),
            None, iterationsCount=self.ransac_iterations,
            reprojectionError=self.reprojection_inlier_threshold, confidence=self.confidence,
            flags=cv.SOLVEPNP_P3P)

        # refine with iterative PnP using inliers only
        if succ and len(inliers) >= 6:
            succ, rvec, tvec, _ = cv.solvePnPGeneric(xyz_0[inliers], pts1[inliers], K1.numpy(
            ), None, useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv.SOLVEPNP_ITERATIVE)
            rvec = rvec[0]
            tvec = tvec[0]

        # avoid degenerate solutions
        if succ:
            if np.linalg.norm(tvec) > 1000:
                succ = False

        if succ:
            R, _ = cv.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
        else:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = []

        return R, t, len(inliers)


class ProcrustesSolver:
    '''Estimate relative pose (metric) using 3D-3D correspondences'''

    def __init__(self, cfg):

        # Procrustes RANSAC parameters
        self.ransac_max_corr_distance = cfg.PROCRUSTES.MAX_CORR_DIST
        self.refine = cfg.PROCRUSTES.REFINE

    def estimate_pose(self, pts0, pts1, data):
        # uses nearest neighbour
        pts0 = np.int32(pts0)
        pts1 = np.int32(pts1)

        if len(pts0) < 3:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # get depth at correspondence points
        depth_0, depth_1 = data['depth0'].squeeze(0), data['depth1'].squeeze(0)
        depth_pts0 = depth_0[pts0[:, 1], pts0[:, 0]]
        depth_pts1 = depth_1[pts1[:, 1], pts1[:, 0]]

        # remove invalid pts (depth == 0)
        valid = (depth_pts0 > depth_0.min()) * (depth_pts1 > depth_1.min())
        if valid.sum() < 3:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        pts0 = pts0[valid]
        pts1 = pts1[valid]
        depth_pts0 = depth_pts0[valid]
        depth_pts1 = depth_pts1[valid]

        # backproject points to 3D in each sensors' local coordinates
        K0 = data['K_color0'].squeeze(0)
        K1 = data['K_color1'].squeeze(0)
        xyz_0 = backproject_3d(pts0, depth_pts0, K0)
        xyz_1 = backproject_3d(pts1, depth_pts1, K1)

        # create open3d point cloud objects and correspondences idxs
        pcl_0 = o3d.geometry.PointCloud()
        pcl_0.points = o3d.utility.Vector3dVector(xyz_0)
        pcl_1 = o3d.geometry.PointCloud()
        pcl_1.points = o3d.utility.Vector3dVector(xyz_1)
        corr_idx = np.arange(pts0.shape[0])
        corr_idx = np.tile(corr_idx.reshape(-1, 1), (1, 2))
        corr_idx = o3d.utility.Vector2iVector(corr_idx)

        # obtain relative pose using procrustes
        ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria()
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcl_0, pcl_1, corr_idx, self.ransac_max_corr_distance, criteria=ransac_criteria)
        inliers = int(res.fitness * np.asarray(pcl_1.points).shape[0])

        # refine with ICP
        if self.refine:
            # first, backproject both (whole) point clouds
            vv, uu = np.mgrid[0:depth_0.shape[0], 0:depth_1.shape[1]]
            uv_coords = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

            valid = depth_0.reshape(-1) > 0
            xyz_0 = backproject_3d(uv_coords[valid], depth_0.reshape(-1)[valid], K0)

            valid = depth_1.reshape(-1) > 0
            xyz_1 = backproject_3d(uv_coords[valid], depth_1.reshape(-1)[valid], K1)

            pcl_0 = o3d.geometry.PointCloud()
            pcl_0.points = o3d.utility.Vector3dVector(xyz_0)
            pcl_1 = o3d.geometry.PointCloud()
            pcl_1.points = o3d.utility.Vector3dVector(xyz_1)

            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-4,
                                                                             relative_rmse=1e-4,
                                                                             max_iteration=30)

            res = o3d.pipelines.registration.registration_icp(pcl_0,
                                                              pcl_1,
                                                              self.ransac_max_corr_distance,
                                                              init=res.transformation,
                                                              criteria=icp_criteria)

        R = res.transformation[:3, :3]
        t = res.transformation[:3, -1].reshape(3, 1)
        inliers = int(res.fitness * np.asarray(pcl_1.points).shape[0])
        return R, t, inliers
