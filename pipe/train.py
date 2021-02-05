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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2path(v):
    return constants.data_path / v


def parse_arguments(str2bool):
    parser = argparse.ArgumentParser()

    # trainer
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--metric", type=str, default="multilabel_auc_macro")
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold to use. -1 to run 5-fold cross-validation",
    )

    # architecture
    parser.add_argument("--arch", type=str)

    # data loader
    parser.add_argument("--train_data", type=str2path)
    parser.add_argument("--test_data", type=str2path)
    parser.add_argument("--sz", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--auto_batch_size", type=str, default=None)

    # optimizer
    parser.add_argument("--opt", type=str, default="sam")
    parser.add_argument("--loss", type=str, default="bce_with_logits")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--mom", type=float, default=0.9)

    # lr scheduler
    parser.add_argument(
        "--auto_lr", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument("--sched", type=str, default="plateau")

    # callbacks
    # parser.add_argument("--patience", type=int)
    # parser.add_argument("--decay_rate", "--dr", type=float)

    # miscellaneous
    parser.add_argument("--verbose", type=bool)

    return parser.parse_args()


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
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
            ),
            albumentations.OneOf(
                [
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
    print("\nCurrent config:\n")
    [print(f"{k}: {v}") for k, v in vars(hparams).items()]
    print("\n")

    dm = data.LitDataModule(
        data_path=constants.data_path,
        batch_size=hparams.batch_size,
        fold=hparams.fold,
        train_image_path=hparams.train_data,
        valid_image_path=hparams.train_data,
        test_image_path=hparams.test_data,
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

    trainer = pl.Trainer(
        gpus=1,
        auto_lr_find=hparams.auto_lr,
        auto_scale_batch_size=hparams.auto_batch_size,
        max_epochs=hparams.epochs,
        callbacks=[checkpoint_callback],
    )
    if hparams.auto_lr or hparams.auto_batch_size:
        print("\nTuning...")
        trainer.tune(model, dm)

    # train and validate model
    trainer.fit(model, dm)
    valid_metric = model.best_valid_metric
    train_metric = model.best_train_metric

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

    return train_metric, valid_metric


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


if __name__ == "__main__":
    hparams = parse_arguments(str2bool)

    if hparams.fold == -1:
        valid_scores = []
        train_scores = []
        for current_fold in range(5):
            hparams.fold = current_fold
            train_score, valid_score = run(hparams=hparams)
            valid_scores.append(valid_score)
            train_scores.append(train_score)

        cv_metric = np.mean(valid_scores)
        train_metric = np.mean(train_scores)
        print(
            f"\n{hparams.metric} // Train: {train_metric:.4f}, CV: {cv_metric:.4f}"
        )
        with open(
            constants.metrics_path
            / f"arch={hparams.arch}_sz={hparams.sz}.metric",
            "w",
        ) as f:
            f.write(
                f"{hparams.metric} // Train: {train_metric:.4f}, CV: {cv_metric:.4f}"
            )
    else:
        train_metric, valid_metric = run(hparams=hparams)
        print(
            f"\nBest {hparams.metric}: Train {train_metric:.4f}, Valid: {valid_metric:.4f}"
        )
