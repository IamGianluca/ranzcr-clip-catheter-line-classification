import argparse

import albumentations
import albumentations.augmentations.transforms as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import transforms
from pytorch_lightning import callbacks

from ml import classification, data
from pipe import constants


def run(hparams: argparse.Namespace):
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

    mean = 0.485
    std = 0.2295
    train_augmentation = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(
                hparams.sz, hparams.sz, scale=(0.9, 1), p=1
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7
            ),
            albumentations.OneOf(
                [
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(
                        num_steps=5, distort_limit=1.0
                    ),
                    albumentations.ElasticTransform(alpha=3),
                ],
                p=0.2,
            ),
            albumentations.OneOf(
                [
                    albumentations.GaussNoise(var_limit=[10, 50]),
                    albumentations.GaussianBlur(),
                    albumentations.MotionBlur(),
                    albumentations.MedianBlur(),
                ],
                p=0.2,
            ),
            albumentations.OneOf(
                [
                    albumentations.JpegCompression(),
                    albumentations.Downscale(scale_min=0.1, scale_max=0.15),
                ],
                p=0.2,
            ),
            albumentations.Cutout(
                max_h_size=int(hparams.sz * 0.1),
                max_w_size=int(hparams.sz * 0.1),
                num_holes=5,
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
        batch_size=hparams.batch_size,
        fold=hparams.fold,
        sz=hparams.sz,
        train_augmentation=train_augmentation,
        valid_augmentation=valid_augmentation,
        test_augmentation=test_augmentation,
    )
    dm.setup()

    model = classification.LitClassifier(
        target_cols=target_cols,
        in_channels=1,
        num_classes=len(target_cols),
        **vars(hparams),
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="valid_metric",
        mode="max",
        dirpath=constants.models_path,
        filename=f"arch={hparams.arch}_sz={hparams.sz}_fold={hparams.fold}",
        save_weights_only=True,
    )
    earlystopping_callback = callbacks.EarlyStopping(
        monitor="valid_metric",
        min_delta=0.0,
        patience=5,
        verbose=False,
        mode="max",
        strict=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        auto_lr_find=hparams.auto_lr,
        auto_scale_batch_size=hparams.auto_batch_size,
        max_epochs=hparams.epochs,
        callbacks=[checkpoint_callback, earlystopping_callback],
    )
    if hparams.auto_lr or hparams.auto_batch_size:
        print("Tuning...")
        trainer.tune(model, dm)

    # train and validate model
    trainer.fit(model, dm)

    # TODO: load best model, not just the latest
    fname = f"arch={hparams.arch}_sz={hparams.sz}_fold={hparams.fold}.csv"
    oof_predictions_fpath = constants.submissions_path / f"oof/{fname}"
    create_submission(
        hparams=hparams,
        target_cols=target_cols,
        dm=dm.val_dataloader,
        model=model,
        is_oof=True,
        fpath=oof_predictions_fpath,
    )

    submission_fpath = constants.submissions_path / fname
    create_submission(
        hparams=hparams,
        target_cols=target_cols,
        dm=dm.test_dataloader,
        model=model,
        is_oof=False,
        fpath=submission_fpath,
    )


def create_submission(hparams, target_cols, dm, model, is_oof, fpath):
    model.freeze()
    model.to("cuda")

    preds = []
    for batch in dm():
        if is_oof:
            x, _ = batch
        else:
            x = batch  # wedon't have targets for test data
        batch_preds = model(x.to("cuda"))
        preds.append(batch_preds.detach().cpu().numpy())
    preds = np.vstack(preds)

    if is_oof:
        result = pd.read_csv(constants.train_folds_fpath)
        result = result.loc[:, ["StudyInstanceUID"] + target_cols + ["kfold"]]
        result = result[result.kfold == hparams.fold]
        result[target_cols] = preds
    else:
        result = pd.read_csv(constants.sample_submission_fpath)
        result[target_cols] = preds

    result.to_csv(fpath, index=False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--opt", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--sz", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--auto_batch_size", type=str, default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--auto_lr", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="reduce_on_plateau"
    )
    parser.add_argument("--wd", type=float)
    parser.add_argument("--mom", type=float)

    hparams = parser.parse_args()

    run(hparams=hparams)
