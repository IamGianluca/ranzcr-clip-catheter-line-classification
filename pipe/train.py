import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from constants import data_path, metrics_path


def train(verbose: bool = False):
    df = pd.read_csv(data_path / 'train_folds.csv')
    label_cols = df.columns[1:-2]

    cv_scores = []
    for fold in range(df.kfold.max()):
        train, valid = df[df.kfold!=fold], df[df.kfold==fold]
        if verbose:
            print(f"Fold {fold}, {train.shape}, {valid.shape}")

        y_pred = np.tile(train.loc[:, label_cols].mean(), (valid.shape[0], 1))
        y_true = valid.loc[:, label_cols].values
        cv_scores.append([roc_auc_score(y_true=y_true, y_score=y_pred, average="macro")])

    with open(metrics_path / 'model_one_cv.metric', 'w') as f:
        f.write(f"AUC: {np.mean(cv_scores):.3f}")


if __name__ == '__main__':
    train()