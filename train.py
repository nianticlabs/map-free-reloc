import argparse
import os
from pathlib import Path

# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from config.default import cfg
from config.utils import config_merge_from_file
from lib.datasets.datamodules import DataModule
from lib.models.regression.model import RegressionModel, RegressionMultiFrameModel


def main(args):
    global cfg
    cfg = config_merge_from_file(cfg=cfg, path_to_config=args.dataset_config)
    cfg = config_merge_from_file(cfg=cfg, path_to_config=args.config)

    pl.seed_everything(0)

    datamodule = DataModule(cfg)
    if cfg.MODEL == 'RegressionMultiFrame':
        model = RegressionMultiFrameModel(cfg)
    elif cfg.MODEL == 'Regression':
        model = RegressionModel(cfg)
    else:
        raise NotImplementedError(f'Invalid model {cfg.MODEL}')

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
    trainer = pl.Trainer(devices=1,
                         log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
                         val_check_interval=cfg.TRAINING.VAL_INTERVAL,
                         limit_val_batches=cfg.TRAINING.VAL_BATCHES,
                         max_epochs=cfg.TRAINING.EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_callback, lr_monitoring_callback, epochend_callback],
                         num_sanity_val_steps=1,
                         gradient_clip_val=cfg.TRAINING.GRAD_CLIP)
                         # track_grad_norm=-1) # TODO: put back the equivalent!

    trainer.fit(model, datamodule, ckpt_path=args.resume)


if __name__ == '__main__':
    """
    Single-frame query:
    ```shell
    python3 train.py \
      config/regression/mapfree/3d3d.yaml config/mapfree.yaml
    ```
    
    Multi-frame query:
    ```shell
    python3 train.py \
      config/regression/mapfree/multiframe/3d3d_multi.yaml \
      config/mapfree.yaml \
      config/mapfree_multi.yaml
    ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', action='append', help='path to config file')
    parser.add_argument('dataset_config', nargs='+', action='store',
                        help='path to dataset config file')
    parser.add_argument('--config', action='append', dest='config',
                        help='one more path to a config file')
    parser.add_argument('--dataset-config', '--dataset_config', action='append',
                        dest='dataset_config', help='one more path to a dataset config file')
    parser.add_argument('--experiment', help='experiment name', default='default')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    args = parser.parse_args()

    assert isinstance(args.config, (list, tuple, str, Path))
    # make sure we don't have nested lists by accident
    if isinstance(args.config, (list, tuple)):
        for args_config in args.config:
            assert isinstance(args_config, (str, Path))

    assert isinstance(args.dataset_config, (list, tuple, str, Path))
    # make sure we don't have nested lists by accident
    if isinstance(args.dataset_config, (list, tuple)):
        for args_dataset_config in args.dataset_config:
            assert isinstance(args_dataset_config, (str, Path))

    main(args)
