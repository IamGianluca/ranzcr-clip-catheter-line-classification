import argparse

import albumentations.augmentations.transforms as A
from pipe import constants
import pytorch_lightning as pl
from albumentations.core import composition
from albumentations.pytorch import transforms
from pytorch_lightning.core.saving import load_hparams_from_yaml

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
        constants.submissions_path / f"oof/submission_{fold}.csv"
    )

    model = classification.LitClassifier(
        target_cols=target_cols,
        sample_submission_fpath=sample_submission_fpath,
        submission_fpath=submission_fpath,
        in_channels=1,
        num_classes=len(target_cols),
        **vars(hparams),
    )
    trainer = pl.Trainer(gpus=1, max_epochs=hparams.epochs)
    trainer.fit(model, dm)

    # create predictions for validation samples
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--verbose", type=bool)

    args = parser.parse_args()

    run(fold=args.fold, verbose=args.verbose)
