import torch
import pytorch_lightning as pl

from lib.models.regression.aggregator import *
from lib.models.regression.head import *
from lib.models.regression.encoder.resnet import ResNet
from lib.models.regression.encoder.resunet import ResUNet

from lib.utils.loss import *
from lib.utils.metrics import pose_error_torch, error_auc, A_metrics


class RegressionModel(pl.LightningModule):
    """Regresses Relative Pose between a pair of images"""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # initialise encoder
        try:
            encoder = eval(cfg.ENCODER.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid encoder {cfg.ENCODER.TYPE}')
        self.encoder = encoder(cfg.ENCODER)

        # aggregator
        try:
            aggregator = eval(cfg.AGGREGATOR.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid aggregator {cfg.AGGREGATOR.TYPE}')
        self.aggregator = aggregator(cfg.AGGREGATOR, self.encoder.num_out_layers)

        # head
        try:
            head = eval(cfg.HEAD.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid head {cfg.HEAD.TYPE}')
        self.head = head(cfg, self.aggregator.num_out_layers)

        # initialise loss function
        try:
            self.rot_loss = eval(cfg.TRAINING.ROT_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid rotation loss {cfg.TRAINING.ROT_LOSS}')
        try:
            self.trans_loss = eval(cfg.TRAINING.TRANS_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid translation loss {cfg.TRAINING.TRANS_LOSS}')

        # set loss function weights
        # if LAMBDA is 0., use learnable weights from
        # Geometric loss functions for camera pose regression with deep learning (Kendal & Cipolla)
        self.LAMBDA = cfg.TRAINING.LAMBDA
        if cfg.TRAINING.LAMBDA == 0.:
            self.s_r = torch.nn.Parameter(torch.zeros(1))
            self.s_t = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data):
        vol0 = self.encoder(data['image0'])
        vol1 = self.encoder(data['image1'])

        global_volume = self.aggregator(vol0, vol1)
        R, t = self.head(global_volume, data)
        data['R'] = R
        data['t'] = t
        data['inliers'] = 0
        return R, t

    def loss_fn(self, data):
        R_loss = self.rot_loss(data)
        t_loss = self.trans_loss(data)

        if self.LAMBDA == 0:
            # PoseNet (Kendal & Cipolla) lernable loss scaling
            loss = R_loss * torch.exp(-self.s_r) + t_loss * torch.exp(-self.s_t) + self.s_r + self.s_t
        else:
            loss = R_loss + self.LAMBDA * t_loss

        return R_loss, t_loss, loss

    def training_step(self, batch, batch_idx):
        self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)

        self.log('train/R_loss', R_loss)
        self.log('train/t_loss', t_loss)
        self.log('train/loss', loss)
        if self.LAMBDA == 0.:
            self.log('train/s_R', self.s_r)
            self.log('train/s_t', self.s_t)
        return loss

    def validation_step(self, batch, batch_idx):
        Tgt = batch['T_0to1']
        R, t = self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)

        # validation metrics
        outputs = pose_error_torch(R, t, Tgt, reduce=None)
        outputs['R_loss'] = R_loss
        outputs['t_loss'] = t_loss
        outputs['loss'] = loss
        return outputs

    def validation_epoch_end(self, outputs):
        # aggregates metrics/losses from all validation steps
        aggregated = {}
        for key in outputs[0].keys():
            aggregated[key] = torch.stack([x[key] for x in outputs])

        # compute stats
        median_t_ang_err = aggregated['t_err_ang'].median()
        median_t_scale_err = aggregated['t_err_scale'].median()
        median_t_euclidean_err = aggregated['t_err_euc'].median()
        median_R_err = aggregated['R_err'].median()
        mean_R_loss = aggregated['R_loss'].mean()
        mean_t_loss = aggregated['t_loss'].mean()
        mean_loss = aggregated['loss'].mean()

        # a1, a2, a3 metrics of the translation vector norm
        a1, a2, a3 = A_metrics(aggregated['t_err_scale_sym'])

        # compute AUC of Euclidean translation error for 10cm, 50cm and 1m thresholds
        AUC_euc_10, AUC_euc_50, AUC_euc_100 = error_auc(
            aggregated['t_err_euc'].view(-1).detach().cpu().numpy(),
            [0.1, 0.5, 1.0]).values()

        # compute AUC of pose error (max of rot and t ang. error) for 5, 10 and 20 degrees thresholds
        pose_error = torch.maximum(
            aggregated['t_err_ang'].view(-1),
            aggregated['R_err'].view(-1)).detach().cpu()
        AUC_pos_5, AUC_pos_10, AUC_pos_20 = error_auc(pose_error.numpy(), [5, 10, 20]).values()

        # compute AUC of rotation error 5, 10 and 20 deg thresholds
        rot_error = aggregated['R_err'].view(-1).detach().cpu()
        AUC_rot_5, AUC_rot_10, AUC_rot_20 = error_auc(rot_error.numpy(), [5, 10, 20]).values()

        # compute AUC of translation angle error 5, 10 and 20 deg thresholds
        t_ang_error = aggregated['t_err_ang'].view(-1).detach().cpu()
        AUC_tang_5, AUC_tang_10, AUC_tang_20 = error_auc(t_ang_error.numpy(), [5, 10, 20]).values()

        # log stats
        self.log('val_loss/R_loss', mean_R_loss)
        self.log('val_loss/t_loss', mean_t_loss)
        self.log('val_loss/loss', mean_loss)
        self.log('val_metrics/t_ang_err', median_t_ang_err)
        self.log('val_metrics/t_scale_err', median_t_scale_err)
        self.log('val_metrics/t_euclidean_err', median_t_euclidean_err)
        self.log('val_metrics/R_err', median_R_err)
        self.log('val_auc/euc_10', AUC_euc_10)
        self.log('val_auc/euc_50', AUC_euc_50)
        self.log('val_auc/euc_100', AUC_euc_100)
        self.log('val_auc/pose_5', AUC_pos_5)
        self.log('val_auc/pose_10', AUC_pos_10)
        self.log('val_auc/pose_20', AUC_pos_20)
        self.log('val_auc/rot_5', AUC_rot_5)
        self.log('val_auc/rot_10', AUC_rot_10)
        self.log('val_auc/rot_20', AUC_rot_20)
        self.log('val_auc/tang_5', AUC_tang_5)
        self.log('val_auc/tang_10', AUC_tang_10)
        self.log('val_auc/tang_20', AUC_tang_20)
        self.log('val_t_scale/a1', a1)
        self.log('val_t_scale/a2', a2)
        self.log('val_t_scale/a3', a3)

        return mean_loss

    def configure_optimizers(self):
        tcfg = self.cfg.TRAINING
        opt = torch.optim.Adam(self.parameters(), lr=tcfg.LR, eps=1e-6)
        if tcfg.LR_STEP_INTERVAL:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, tcfg.LR_STEP_INTERVAL, tcfg.LR_STEP_GAMMA)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        return opt
