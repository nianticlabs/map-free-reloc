import torch.cuda

from lib.models.regression.model import RegressionModel
from lib.models.regression.model import RegressionMultiFrameModel
from lib.models.matching.model import FeatureMatchingModel


def build_model(cfg, checkpoint=''):
    if cfg.MODEL == 'FeatureMatching':
        return FeatureMatchingModel(cfg)
    elif cfg.MODEL == 'Regression':
        model = RegressionModel.load_from_checkpoint(checkpoint, cfg=cfg) if \
            checkpoint is not '' else RegressionModel(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    elif cfg.MODEL == 'RegressionMultiFrame':
        model = RegressionMultiFrameModel.load_from_checkpoint(checkpoint, cfg=cfg) if \
            checkpoint is not '' else RegressionMultiFrameModel(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    else:
        raise NotImplementedError()
