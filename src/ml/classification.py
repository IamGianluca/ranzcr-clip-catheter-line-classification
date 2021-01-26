from pathlib import Path
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss import loss_factory
from .metrics import metric_factory
from .optim import optimizer_factory


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        target_cols: List[str],
        sample_submission_fpath: Path,
        submission_fpath: Path,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # TODO: create model factory and incapsulate changes required to
        # work with inputs of different in_channels, num_classes,
        # non-linearity, etc.
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

    def configure_optimizers(self):
        config = dict()
        optimizer = optimizer_factory(
            name=self.hparams.opt, parameters=self.parameters()
        )
        config["optimizer"] = optimizer

        if True:
            config["lr_scheduler"] = ReduceLROnPlateau(
                optimizer, mode="max", patience=3, verbose=True
            )
            config["monitor"] = "valid_metric"
        return config

    def loss_function(self, y_pred, y_true):
        y_true = y_true.float()  # TODO: find a way to avoid this

        loss_fn = loss_factory(name=self.hparams.loss)
        loss = loss_fn(y_pred, y_true.view(-1))
        return loss

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
        except ValueError:  # bs to small
            train_metric = 0.0

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
        except ValueError:  # bs to small
            valid_metric = 0

        if self.current_epoch >= 1:
            try:
                train_loss = self.trainer.callback_metrics["train_loss"]
                train_metric = self.trainer.callback_metrics["train_metric"]
                self.trainer.progress_bar_callback.main_progress_bar.write(
                    f"Epoch {self.current_epoch} // train loss: {train_loss:.4f}, train metric: {train_metric:.4f}, valid loss: {valid_loss:.4f}, valid metric: {valid_metric:.4f}"
                )
            except (KeyError, AttributeError):
                # these errors occurs when in "tuning" mode (find optimal lr)
                pass

        self.log(
            "valid_loss",
            valid_loss,
        )
        self.log("valid_metric", valid_metric)

    def test_step(self, batch, batch_idx):
        x_test = batch
        y_pred = self(x_test)
        return {"y_pred": y_pred}

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([out["y_pred"] for out in outputs])

        submission = pd.read_csv(self.hparams.sample_submission_fpath)
        submission[self.hparams.target_cols] = y_pred.detach().cpu().numpy()
        submission.to_csv(self.hparams.submission_fpath, index=False)
