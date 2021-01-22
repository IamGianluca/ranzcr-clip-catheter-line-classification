import argparse

import numpy as np
import joblib
import pandas as pd
import pytorch_lightning as pl
from albumentations.core import composition
import albumentations.augmentations.transforms as A
from albumentations.pytorch import transforms
from pytorch_lightning.core.saving import load_hparams_from_yaml

from ml import constants
from ml import data
from ml import classification
from ml import utils


def run(fold, verbose: bool = False):
    df = pd.read_csv(constants.data_path / "train_folds.csv")

    # organize feature names in lists we can reuse later on
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

    # training data is where kfold is not equal to provided fold
    # also, note we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column(s) from dataframe and convert it to a numpy array
    params = load_hparams_from_yaml(constants.params_fpath)
    hparams = utils.dict_to_args(params["train_resnet_18"])

    train_augmentation = composition.Compose(
        [A.Resize(hparams.sz, hparams.sz), transforms.ToTensorV2()]
    )
    valid_augmentation = composition.Compose(
        [A.Resize(hparams.sz, hparams.sz), transforms.ToTensorV2()]
    )
    test_augmentation = composition.Compose(
        [A.Resize(hparams.sz, hparams.sz), transforms.ToTensorV2()]
    )

    dm = data.RazncrDataModule(
        data_path=constants.data_path,
        batch_size=hparams.bs,
        fold=fold,
        train_augmentation=train_augmentation,
        valid_augmentation=valid_augmentation,
        test_augmentation=test_augmentation,
    )
    dm.setup()

    model = classification.LitClassifier(
        in_channels=1, num_classes=len(target_cols), hparams=hparams
    )
    trainer = pl.Trainer(gpus=1, max_epochs=2)
    trainer.fit(model, dm)

    # create predictions for validation samples
    trainer.test(datamodule=dm)
    # y_pred = clf.predict(x_valid)

    # # calculate and print evaluation metric
    # cv_score = metrics.roc_auc_score(
    #     y_true=y_valid, y_score=y_pred, average="macro"
    # )
    # message = f"Fold={fold}, AUC: {cv_score:.3f}"
    # print(message)

    # with open(constants.metrics_path / "model_one_cv.metric", "w") as f:
    #     f.write(message)

    # # save the model
    # joblib.dump(clf, constants.models_path / f"dt{model}_{fold}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--verbose", type=bool)

    args = parser.parse_args()

    run(fold=args.fold, verbose=args.verbose)