import constants
import pandas as pd
from sklearn.model_selection import GroupKFold

from pipe import constants


def split():
    df = pd.read_csv(constants.data_path / "train.csv")

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    x = df.drop(constants.target_cols, axis=1)
    y = df.loc[:, constants.target_cols]
    groups = df.PatientID

    kf = GroupKFold(n_splits=5)
    for fold, (_, val_idx) in enumerate(
        kf.split(X=x, y=y, groups=groups)
    ):
        df.loc[val_idx, "kfold"] = fold

    print(df.groupby("kfold").mean())

    df.to_csv(constants.data_path / "train_folds.csv", index=False)


if __name__ == "__main__":
    split()
