import argparse

import albumentations.augmentations.transforms as A
from albumentations.core.composition import OneOf
from numpy.lib.function_base import kaiser
from pytorch_lightning import callbacks
from pipe import constants
import pytorch_lightning as pl
import albumentations
from albumentations.pytorch import transforms
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from ml import classification, data, utils


def run(fold: int, verbose: bool = False):
    target_cols = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]

    # drop the label column(s) from dataframe and convert it to a numpy array
    params = load_hparams_from_yaml(constants.params_fpath)
    hparams = utils.dict_to_args(params["train_resnet18_128"])

    mean = 0.485
    std = 0.2295
    train_augmentation = albumentations.Compose(
        [
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
            ),
            albumentations.OneOf(
                [
                    # albumentations.RandomGamma(gamma_limit=(90, 110)),
                    albumentations.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1
                    ),
                ],
                p=0.5,
            ),
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
            transforms.ToTensorV2(),
        ]
    )
    valid_augmentation = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
            transforms.ToTensorV2(),
        ]
    )
    test_augmentation = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            ),
            transforms.ToTensorV2(),
        ]
    )

    # print config to terminal
    print("Current config:\n")
    [print(f"{k}: {v}") for k, v in vars(hparams).items()]
    print("\n")

    dm = data.LitDataModule(
        data_path=constants.data_path,
        batch_size=hparams.bs,
        fold=fold,
        train_augmentation=train_augmentation,
        valid_augmentation=valid_augmentation,
        test_augmentation=test_augmentation,
    )
    dm.setup()

    sample_submission_fpath = constants.data_path / "sample_submission.csv"
    submission_fpath = (
        constants.submissions_path
        / f"oof/arch={hparams.arch}_sz={hparams.sz}_fold={fold}.csv"
    )

    model = classification.LitClassifier(
        target_cols=target_cols,
        sample_submission_fpath=sample_submission_fpath,
        submission_fpath=submission_fpath,
        in_channels=1,
        num_classes=len(target_cols),
        **vars(hparams),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_metric",
        mode="max",
        dirpath=constants.models_path,
        filename=f"arch={hparams.arch}_sz={hparams.sz}_fold={fold}",
        save_weights_only=True,
    )

    auto_lr_find = False
    trainer = pl.Trainer(
        gpus=1,
        auto_lr_find=auto_lr_find,
        max_epochs=hparams.epochs,
        callbacks=[checkpoint_callback],
    )
    if auto_lr_find:
        trainer.tune(model, dm)

    trainer.fit(model, dm)

    # create predictions for validation samples
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--verbose", type=bool)

    args = parser.parse_args()

    run(fold=args.fold, verbose=args.verbose)
