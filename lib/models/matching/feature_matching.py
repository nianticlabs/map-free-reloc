import numpy as np
import cv2 as cv


class PrecomputedMatching:
    '''Get correspondences from pre-computed file'''

    def __init__(self, cfg):
        # Scannet correspondences are stored in a single file, pointed by MATCHES_FILE_PATH
        # 7Scenes correspondences are split in a file per scene and dependent on the pairs.
        # The 7Scenes file pattern (including {scene_id} and {test_pairs} tags) is stored in MATCHES_FILE_PATH

        self.correspondences = None
        self.debug = cfg.DEBUG

        # If there is a pattern, save that string pattern, and will load correspondences once the scene_id is defined
        if '{' in cfg.MATCHES_FILE_PATH:
            self.matches_file_path = cfg.MATCHES_FILE_PATH
            self.scene_id = None
            self.pairs_txt = cfg.DATASET.PAIRS_TXT.TEST
        else:
            self.load_correspondences(cfg.MATCHES_FILE_PATH)

    def load_correspondences(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.correspondences = data['correspondences'].astype(np.float32)

    def get_correspondences(self, data):
        # Check if loaded scene_id is still valid (in the case where correspondences are stored over multiple files)
        # If not, load the correct scene_id correspondences
        if hasattr(self, 'scene_id'):
            if self.scene_id != data['scene_id'][0]:
                self.scene_id = data['scene_id'][0]
                scene_root = data['scene_root'][0]
                matches_fpath = self.matches_file_path.format(
                    scene_root=scene_root, pairs_txt=self.pairs_txt)
                self.load_correspondences(matches_fpath)

        # get correspondences for the given pair
        pair_id = data['pair_id'].item()
        corr = self.correspondences[pair_id]

        # remove nan's (filler)
        corr = corr[~np.isnan(corr)].reshape(-1, 4)
        if len(corr) > 0:
            pts1, pts2 = corr[:, :2], corr[:, 2:]
        else:
            pts1 = pts2 = np.array([])

        return pts1, pts2


class SIFTMatching:
    def __init__(self, cfg):

        # SIFT parameters
        self.ratio_threshold = cfg.SIFT.RATIO_THRESHOLD
        self.sift = cv.SIFT_create(cfg.SIFT.NUM_FEATURES)
        self.debug = cfg.DEBUG

    def transform_grayscale(self, img):
        img = img.permute(1, 2, 0).numpy()
        img = (255 * img).astype(np.uint8)
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return img_gray

    def root_sift(self, descs):
        '''Apply the Hellinger kernel by first L1-normalizing, taking the square-root, and then L2-normalizing'''

        eps = 1e-7
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return descs

    def get_correspondences(self, data):
        # get grayscale images
        img0 = self.transform_grayscale(data['image0'].squeeze(0))
        img1 = self.transform_grayscale(data['image1'].squeeze(0))

        # get SIFT key points and descriptors
        kp0, des0 = self.sift.detectAndCompute(img0, None)
        kp1, des1 = self.sift.detectAndCompute(img1, None)

        # Apply normalisation (rootSIFT)
        des0, des1 = self.root_sift(des0), self.root_sift(des1)

        # Get matches using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des0, des1, k=2)

        pts1 = []
        pts2 = []
        good_matches = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < self.ratio_threshold * n.distance:
                pts2.append(kp1[m.trainIdx].pt)
                pts1.append(kp0[m.queryIdx].pt)
                good_matches.append(m)

        pts1 = np.float32(pts1).reshape(-1, 2)
        pts2 = np.float32(pts2).reshape(-1, 2)

        # plot results (DEBUG)
        if self.debug:
            img_matches = np.empty(
                (max(img0.shape[0],
                     img1.shape[0]),
                 img1.shape[1] + img1.shape[1],
                 3),
                dtype=np.uint8)
            cv.drawMatches(img0, kp0, img1, kp1, good_matches, img_matches,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            data['debug_img_matches'] = img_matches
        return pts1, pts2
