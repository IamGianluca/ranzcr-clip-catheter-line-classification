import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from constants import data_path, metrics_path


def train():
    df = pd.read_csv(data_path / 'train_folds.csv')
    label_cols = df.columns[1:-2]

    cv_scores = []
    for fold in range(5):
        train, valid = df[df.kfold!=fold], df[df.kfold==fold]
        y_true = valid.loc[:, label_cols].values
        y_pred = np.tile(train.loc[:, label_cols].mean().values, (valid.shape[0], 1))
        cv_scores.append(np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(11)]))

    with open(metrics_path / 'model_one_cv.metric', 'w') as f:
        f.write(f"AUC: {np.mean(cv_scores)}")


if __name__ == '__main__':
    train()