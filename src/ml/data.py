from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# sometimes, you will have images without an ending bit; this
# takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, image_paths, targets, augmentation):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentation = augmentation
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = np.array(Image.open(self.image_paths[item]))

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        if self.targets is not None:
            return image, self.targets[item]
        else:
            return image


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        batch_size,
        fold: int,
        train_augmentation=None,
        valid_augmentation=None,
        test_augmentation=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.fold = fold
        self.train_augmentation = train_augmentation
        self.valid_augmentation = valid_augmentation
        self.test_augmentation = test_augmentation
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self):
        df = pd.read_csv(self.data_path / "train_folds.csv")
        target_cols = df.columns[1:-2]
        df_train = df[df.kfold != self.fold]
        df_valid = df[df.kfold == self.fold]

        # TODO: we should pass the path(s) as an argument to the LitClassifier
        # constructor. In this way, this class will be more generalizable
        train_image_paths = [
            self.data_path / f"train_128/{x}.jpg"
            for x in df_train.StudyInstanceUID.values
        ]
        valid_image_paths = [
            self.data_path / f"train_128/{x}.jpg"
            for x in df_valid.StudyInstanceUID.values
        ]
        test_image_paths = [
            x for x in (self.data_path / "test_128").iterdir() if x.is_file()
        ]

        train_targets = df_train.loc[:, target_cols].values
        valid_targets = df_valid.loc[:, target_cols].values

        self.train_ds = ImageDataset(
            train_image_paths,
            targets=train_targets,
            augmentation=self.train_augmentation,
        )
        self.valid_ds = ImageDataset(
            valid_image_paths,
            targets=valid_targets,
            augmentation=self.valid_augmentation,
        )
        self.test_ds = ImageDataset(
            test_image_paths, targets=None, augmentation=self.test_augmentation
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
