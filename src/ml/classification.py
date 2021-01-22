from typing import List

import torch
from torch import nn
import pytorch_lightning as pl
import torchvision.models as models


class LitClassifier(pl.LightningModule):
    def __init__(self, in_channels: int, num_classes: int, hparams) -> 0:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hparams = hparams

        self.model = models.__dict__[self.hparams.arch](pretrained=True)

        # input has only 1 channel (black-white images) instead of RGB
        self.model.conv1 = nn.Conv2d(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # output for multi-class/multi-label classification problem
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

    def forward(self, x):
        x = self.model(torch.as_tensor(data=x, dtype=torch.float32))
        return x

    def training_step(self, batch, batch_idx):
        x_train, y_train = batch
        y_pred = self(x_train).view(-1)
        train_loss = self.loss_function(y_pred=y_pred, y_true=y_train)
        return {
            "loss": train_loss,
            "y_pred": y_pred,
            "y_true": y_train,
        }

    def training_epoch_end(self, outputs: List):
        train_loss = torch.cat(
            [out["loss"].unsqueeze(dim=0) for out in outputs]
        ).mean()
        y_pred = torch.cat([out["y_pred"] for out in outputs], dim=0)
        y_true = torch.cat([out["y_true"] for out in outputs], dim=0)
        train_auc = 0

        self.log("train_loss", train_loss)
        self.log("train_auc", train_auc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def loss_function(self, y_pred, y_true):
        y_true = y_true.float()
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_pred, y_true.view(-1))
        return loss