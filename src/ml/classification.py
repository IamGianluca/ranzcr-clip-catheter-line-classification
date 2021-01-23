from typing import List

import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch import nn

from .loss import loss_factory
from .metrics import metric_factory
from .optim import optimizer_factory


class LitClassifier(pl.LightningModule):
    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = models.__dict__[self.hparams.arch](pretrained=True)

        # input has only 1 channel (black-white images) instead of RGB
        self.model.conv1 = nn.Conv2d(
            self.hparams.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # output for multi-class/multi-label classification problem
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.hparams.num_classes)

    def forward(self, x):
        x = self.model(torch.as_tensor(data=x, dtype=torch.float32))
        return x

    def training_step(self, batch, batch_idx):
        x_train, y_train = batch
        y_pred = self(x_train)
        train_loss = self.loss_function(
            y_pred=y_pred.view(-1), y_true=y_train.view(-1)
        )
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

        metric = metric_factory(name=self.hparams.metric)
        try:
            train_metric = metric(
                y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            )
        except TypeError:  # sklearn metric, requires dispatching to cpu
            train_metric = metric(
                y_true=y_pred.detach().cpu().numpy(),
                y_pred=y_pred.detach().cpu().numpy(),
            )

        print(f"\nTrain metric: {train_metric}")

        self.log("train_loss", train_loss)
        self.log("train_metric", train_metric)

    def validation_step(self, batch, batch_idx):
        x_valid, y_valid = batch
        y_pred = self(x_valid)
        valid_loss = self.loss_function(
            y_pred=y_pred.view(-1), y_true=y_valid.view(-1)
        )
        return {"valid_loss": valid_loss, "y_pred": y_pred, "y_true": y_valid}

    def validation_epoch_end(self, outputs: List):
        valid_loss = torch.cat(
            [out["valid_loss"].unsqueeze(dim=0) for out in outputs]
        ).mean()
        y_pred = torch.cat([out["y_pred"] for out in outputs], dim=0)
        y_true = torch.cat([out["y_true"] for out in outputs], dim=0)

        metric = metric_factory(name=self.hparams.metric)
        try:
            valid_metric = metric(
                y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            )
        except TypeError:  # sklearn metric, requires dispatching to cpu
            valid_metric = metric(
                y_true=y_pred.detach().cpu().numpy(),
                y_pred=y_pred.detach().cpu().numpy(),
            )

        print(f"\nValid metric: {valid_metric}")

        self.log("valid_loss", valid_loss)
        self.log("valid_metric", valid_metric)

    def configure_optimizers(self):
        opt = optimizer_factory(
            name=self.hparams.opt, parameters=self.parameters()
        )
        return opt

    def loss_function(self, y_pred, y_true):
        y_true = y_true.float()  # TODO: find a way to avoid this

        loss_fn = loss_factory(name=self.hparams.loss)
        loss = loss_fn(y_pred, y_true.view(-1))
        return loss
