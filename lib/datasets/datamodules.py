import torch.utils as utils
from torchvision.transforms import ColorJitter, Grayscale
import pytorch_lightning as pl

from lib.datasets.sampler import RandomConcatSampler
from lib.datasets.scannet import ScanNetDataset
from lib.datasets.sevenscenes import SevenScenesDataset
from lib.datasets.mapfree import MapFreeDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        datasets = {'ScanNet': ScanNetDataset,
                    '7Scenes': SevenScenesDataset,
                    'MapFree': MapFreeDataset}

        assert cfg.DATASET.DATA_SOURCE in datasets.keys(), 'invalid DATA_SOURCE, this dataset is not implemented'
        self.dataset_type = datasets[cfg.DATASET.DATA_SOURCE]

    def get_sampler(self, dataset, reset_epoch=False):
        if self.cfg.TRAINING.SAMPLER == 'scene_balance':
            sampler = RandomConcatSampler(dataset,
                                          self.cfg.TRAINING.N_SAMPLES_SCENE,
                                          self.cfg.TRAINING.SAMPLE_WITH_REPLACEMENT,
                                          shuffle=True,
                                          reset_on_iter=reset_epoch
                                          )
        else:
            sampler = None
        return sampler

    def train_dataloader(self):
        transforms = ColorJitter() if self.cfg.DATASET.AUGMENTATION_TYPE == 'colorjitter' else None
        transforms = Grayscale(
            num_output_channels=3) if self.cfg.DATASET.BLACK_WHITE else transforms

        dataset = self.dataset_type(self.cfg, 'train', transforms=transforms)
        sampler = self.get_sampler(dataset)
        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.TRAINING.BATCH_SIZE,
                                           num_workers=self.cfg.TRAINING.NUM_WORKERS,
                                           sampler=sampler
                                           )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset_type(self.cfg, 'val')
        # Scannet uses scene_balance for validation (resets at each epoch, so val samples are always the same)
        if isinstance(dataset, ScanNetDataset):
            sampler = self.get_sampler(dataset, reset_epoch=True)
        else:
            sampler = None
        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.TRAINING.BATCH_SIZE,
                                           num_workers=self.cfg.TRAINING.NUM_WORKERS,
                                           sampler=sampler,
                                           drop_last=True
                                           )
        return dataloader

    def test_dataloader(self):
        dataset = self.dataset_type(self.cfg, 'test')
        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=1,
                                           num_workers=1,
                                           shuffle=False)
        return dataloader
