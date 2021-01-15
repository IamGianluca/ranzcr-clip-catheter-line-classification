from pathlib import Path

import pandas as pd 
from sklearn.model_selection import KFold


def split():
    path = Path('/home/gianluca/git/kaggle/ranzcr/')
    data_path = path / "data"

    df = pd.read_csv(data_path / "train.csv")

    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)


    kf = KFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv(data_path / "train_folds.csv", index=False)


if __name__== "__main__":
    split()