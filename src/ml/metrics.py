from functools import partial

from sklearn.metrics import roc_auc_score


def metric_factory(name):
    if name == "multilabel_auc_macro":
        return partial(roc_auc_score, average="macro")
