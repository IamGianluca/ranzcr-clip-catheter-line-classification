from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# sometimes, you will have images without an ending bit; this
# takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, image_paths, targets, augmentations):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = np.array(Image.open(self.image_paths[item]))

        if self.augmentations:
            image = self.augmentations(image=image)["image"]

        if self.targets is not None:
            return image, self.targets[item]
        else:
            return image


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_image_paths: List[Path] = None,
        valid_image_paths: List[Path] = None,
        test_image_paths: List[Path] = None,
        train_targets: np.ndarray = None,
        valid_targets: np.ndarray = None,
        train_augmentations=None,
        valid_augmentations=None,
        test_augmentations=None,
    ):
        super().__init__()
        self.train_image_paths = train_image_paths
        self.valid_image_paths = valid_image_paths
        self.test_image_paths = test_image_paths

        self.train_targets = train_targets
        self.valid_targets = valid_targets

        self.train_augmentations = train_augmentations
        self.valid_augmentations = valid_augmentations
        self.test_augmentations = test_augmentations

        self.batch_size = batch_size

    def setup(self):
        if self.train_image_paths:
            self.train_ds = ImageDataset(
                image_paths=self.train_image_paths,
                targets=self.train_targets,
                augmentations=self.train_augmentations,
            )
        if self.valid_image_paths:
            self.valid_ds = ImageDataset(
                image_paths=self.valid_image_paths,
                targets=self.valid_targets,
                augmentations=self.valid_augmentations,
            )
        if self.test_image_paths:
            self.test_ds = ImageDataset(
                image_paths=self.test_image_paths,
                targets=None,
                augmentations=self.test_augmentations,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            drop_last=False,
        )