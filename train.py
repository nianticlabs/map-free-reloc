import argparse
import os
# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.regression.model import RegressionModel


def main(args):
    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    datamodule = DataModule(cfg)
    model = RegressionModel(cfg)

    logger = TensorBoardLogger(save_dir='weights', name=args.experiment)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=5,
        verbose=True,
        monitor='val_loss/loss',
        mode='min'
    )

    epochend_callback = pl.callbacks.ModelCheckpoint(
        filename='e{epoch}-last',
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )

    lr_monitoring_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=1,
                         log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
                         val_check_interval=cfg.TRAINING.VAL_INTERVAL,
                         limit_val_batches=cfg.TRAINING.VAL_BATCHES,
                         max_epochs=cfg.TRAINING.EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_callback, lr_monitoring_callback, epochend_callback],
                         num_sanity_val_steps=0,
                         gradient_clip_val=cfg.TRAINING.GRAD_CLIP,
                         track_grad_norm=-1)

    trainer.fit(model, datamodule, ckpt_path=args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('dataset_config', help='path to dataset config file')
    parser.add_argument('--experiment', help='experiment name', default='default')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    args = parser.parse_args()

    main(args)
