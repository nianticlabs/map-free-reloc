from yacs.config import CfgNode as CN

_CN = CN()

##############  Model    ##############
_CN.MODEL = None  # options: ['Regression', 'FeatureMatching']
_CN.DEBUG = False

# Regression model options
_CN.ENCODER = CN()
_CN.ENCODER.TYPE = None   # options: ['ResNet', 'ResUNet']
_CN.ENCODER.NUM_BLOCKS = None  # # blocks per layer separated by dashes. e.g. 3-3-3
_CN.ENCODER.BLOCK_TYPE = None  # 0:PreactBlock, 1:PreactBlockBottleneck
_CN.ENCODER.NOT_CONCAT = None  # ResUNet option
_CN.ENCODER.NUM_OUT_LAYERS = None  # ResUNet option

_CN.AGGREGATOR = CN()
_CN.AGGREGATOR.TYPE = None  # options: ['CorrelationVolumeWarping', 'CorrelationVolumeWarpingQKV']
_CN.AGGREGATOR.POSITION_ENCODER = None   # True/False. If True adds two channel with average u,v coordinates of warp
_CN.AGGREGATOR.POSITION_ENCODER_IM1 = None   # True/False. If True adds two channel with uniform u,v coordinates of im1
_CN.AGGREGATOR.MAX_SCORE_CHANNEL = None  # True/False. If True adds a channel with max score to global features
_CN.AGGREGATOR.NORMALISE_DOT = False     # True/False. If True normalise features before dot product
_CN.AGGREGATOR.RESIDUAL_ATT = False      # True/False. If True Q,K,V are residuals from features
_CN.AGGREGATOR.CV_OUTLAYERS = 0          # If >0, compresses CorrelationVolume into OutLayers and channel-wise append to Global Volume
_CN.AGGREGATOR.CV_HALF_CHANNELS = False  # If True, computes correlation volume using only half the images feature channels, giving more freedom for the rest
_CN.AGGREGATOR.UPSAMPLE_POS_ENC = 0      # If >0, upsamples positional encoder with number of channels
_CN.AGGREGATOR.DUSTBIN = False           # If True, creates dustbins to assign 'unmatched' features. Also learns a 'dustbin feature' to be used when warping feature maps

_CN.HEAD = CN()
_CN.HEAD.TYPE = None     # options: ['ProcrustesResBlockMLP', 'DirectResBlockMLP']
_CN.BACKPROJECT_ANCHORS = None    # whether to backproject anchors to 3D or assume that HEAD already gives 3D points
_CN.HEAD.ADD_BASIS = False        # if true, add orthonormal basis to MLP anchors, only valid if NUM_PTS=3 or 6
_CN.HEAD.NUM_PTS = 6              # number of points to estimate. 3, 6 or more. (3: predict correspondences to fixed orthonormal-basis, 6: predict full 3D-3D correspondences, even, more than 6: predict overcomplete set)
_CN.HEAD.AVG_POOL = False         # if true, reduce last feature volume to vector using Global Avg. Pool. Otherwise, use ravel()
_CN.HEAD.BATCH_NORM = True        # enable/disable batch-norm for head res-blocks
_CN.HEAD.SEPARATE_SCALE = True    # For QuatDeepResblock: if True, regress scale separately (unitary translation vector (3D) + 1D scale); else, regress scaled translation vector (3D)
                                  # For AngularBinsResblock: if True, regress scale separately (bins for trans. angle + 1D scale); else, regress scaled translation vector

# Feature Matching Options
_CN.FEATURE_MATCHING = None  # options: ['SIFT', 'Precomputed']
_CN.POSE_SOLVER = None  # options: ['EssentialMatrix', 'EssentialMatrixMetric', 'Procrustes', 'PNP']

# SIFT options
_CN.SIFT = CN()
_CN.SIFT.NUM_FEATURES = None
_CN.SIFT.RATIO_THRESHOLD = None

# Pre-computed feature matching options
_CN.MATCHES_FILE_PATH = None    # path to NPY storing the correspondences pre-computed using the learned algorithm

# EMAT RANSAC options
_CN.EMAT_RANSAC = CN()
_CN.EMAT_RANSAC.PIX_THRESHOLD = None
_CN.EMAT_RANSAC.SCALE_THRESHOLD = None
_CN.EMAT_RANSAC.CONFIDENCE = None

# Procrustes RANSAC options
_CN.PROCRUSTES = CN()
_CN.PROCRUSTES.MAX_CORR_DIST = None
_CN.PROCRUSTES.REFINE = False      #refine pose with ICP

# PNP RANSAC options
_CN.PNP = CN()
_CN.PNP.RANSAC_ITER = None
_CN.PNP.REPROJECTION_INLIER_THRESHOLD = None  # pixels
_CN.PNP.CONFIDENCE = None

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.DATA_SOURCE = None # options: ['ScanNet', '7Scenes', 'MapFree']
_CN.DATASET.SCENES = None      # scenes to use (for 7Scenes/MapFree); should be a list []; If none, use all scenes.
_CN.DATASET.DATA_ROOT = None   # path to dataset folder
_CN.DATASET.NPZ_ROOT = None    # path to npz files containing pairs of frame indices per sample
_CN.DATASET.MIN_OVERLAP_SCORE = None  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MAX_OVERLAP_SCORE = None  # discard data with overlap_score > max_overlap_score
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'colorjitter']
_CN.DATASET.BLACK_WHITE = False       # if true, transform images to black & white
_CN.DATASET.PAIRS_TXT = CN()          # Path to text file defining the train/val/test pairs (7Scenes)
_CN.DATASET.PAIRS_TXT.TRAIN = None
_CN.DATASET.PAIRS_TXT.VAL = None
_CN.DATASET.PAIRS_TXT.TEST = None
_CN.DATASET.PAIRS_TXT.ONE_NN = False  # If true, keeps only reference image w/ highest similarity to each query
_CN.DATASET.HEIGHT = None
_CN.DATASET.WIDTH = None
_CN.DATASET.ESTIMATED_DEPTH = None  # Use 'estimated' predictions of depth map, if None uses GT depth map
                                    # For Scannet: path to NPZ storing the depth maps (for a given method); if None use GT depth
                                    # For 7Scenes: suffix to add to depthpath when loading depth maps;  if None use GT depth
                                    # For Mapfree: suffix to add to depthpath when loading depth maps;  if None, no depth 

############# TRAINING #############
_CN.TRAINING = CN()
# Data Loader settings
_CN.TRAINING.BATCH_SIZE = None
_CN.TRAINING.NUM_WORKERS = None
_CN.TRAINING.SAMPLER = None  # options: ['random', 'scene_balance']
_CN.TRAINING.N_SAMPLES_SCENE = None  # if 'scene_balance' sampler, the number of samples to get per scene
_CN.TRAINING.SAMPLE_WITH_REPLACEMENT = None  # if 'scene_balance' sampler, whether to sample with replacement
# Training settings
_CN.TRAINING.LR = None
_CN.TRAINING.LR_STEP_INTERVAL = None
_CN.TRAINING.LR_STEP_GAMMA = None      # multiplicative factor of LR every LR_STEP_ITERATIONS
_CN.TRAINING.VAL_INTERVAL = None
_CN.TRAINING.VAL_BATCHES = None
_CN.TRAINING.LOG_INTERVAL = None
_CN.TRAINING.EPOCHS = None
_CN.TRAINING.GRAD_CLIP = 0.   #  Indicates the L2 norm at which to clip the gradient. Disabled if 0
# Loss settings
_CN.TRAINING.ROT_LOSS = 'rot_frobenius_loss'  # options: ['rot_frobenius_loss', 'rot_l1_loss', 'rot_angle_loss']
_CN.TRAINING.TRANS_LOSS = 'trans_l2_loss'     # options: ['trans_l2_loss', 'trans_ang_loss']
_CN.TRAINING.LAMBDA = 1.0  # scaling term for the translation loss term. If 0.0, learns optimal weighting.



cfg = _CN