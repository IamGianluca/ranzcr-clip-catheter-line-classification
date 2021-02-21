import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import callbacks

from ml import learner, data
from pipe import constants, augmentations


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


def parse_arguments():
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
    parser.add_argument("--aug", type=str)

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
    parser.add_argument("--label_smoothing", type=float, default=0.0)
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

    train_aug, valid_aug, _ = augmentations.augmentations_factory(hparams)

    # print config to terminal
    print("\nCurrent config:\n")
    [print(f"{k}: {v}") for k, v in vars(hparams).items()]
    print("\n")

    # get image paths and targets
    df = pd.read_csv(constants.train_folds_fpath)
    target_cols = df.columns[1:-2]
    df_train = df[df.kfold != hparams.fold].reset_index()
    df_valid = df[df.kfold == hparams.fold].reset_index()

    train_image_paths = [
        hparams.train_data / f"{x}.jpg"
        for x in df_train.StudyInstanceUID.values
    ]
    valid_image_paths = [
        hparams.train_data / f"{x}.jpg"
        for x in df_valid.StudyInstanceUID.values
    ]
    train_targets = df_train.loc[:, target_cols].values
    valid_targets = df_valid.loc[:, target_cols].values

    dm = data.ImageDataModule(
        batch_size=hparams.batch_size,
        # train
        train_image_paths=train_image_paths,
        train_targets=train_targets,
        train_augmentations=train_aug,
        # valid
        valid_image_paths=valid_image_paths,
        valid_targets=valid_targets,
        valid_augmentations=valid_aug,
    )
    dm.setup()

    model = learner.ImageClassifier(
        in_channels=1,
        num_classes=11,
        pretrained=True,
        **vars(hparams),
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="valid_metric",
        mode="max",
        dirpath=constants.models_path,
        filename=f"arch={hparams.arch}_sz={hparams.sz}_fold={hparams.fold}",
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=hparams.precision,
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

    return model.best_train_metric, model.best_valid_metric


if __name__ == "__main__":
    hparams = parse_arguments()

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
