import torch.nn as nn


def loss_factory(name):
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Loss not supported yet.")
