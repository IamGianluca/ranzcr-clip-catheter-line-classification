from typing import List

import pytorch_lightning as pl
import torch
from timm.models import create_model

from .loss import loss_factory
from .metrics import metric_factory
from .optim import optimizer_factory, lr_scheduler_factory


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.best_train_metric = None
        self.best_valid_metric = None

        self.model = create_model(
            model_name=self.hparams.arch,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        x = self.model(torch.as_tensor(data=x))
        return x

    def configure_optimizers(self):
        config = dict()

        # configure optimizer
        optimizer = optimizer_factory(
            params=self.parameters(), hparams=self.hparams
        )
        config["optimizer"] = optimizer

        # configure lr scheduler
        config["lr_scheduler"] = lr_scheduler_factory(
            optimizer=optimizer,
            hparams=self.hparams,
            data_loader=self.train_dataloader(),
        )
        config["monitor"] = "valid_metric"
        return config

    def compute_loss(self, y_hat, y):
        y = y.float()  # TODO: avoid this

        loss_fn = loss_factory(name=self.hparams.loss)
        loss = loss_fn(y_hat.view(-1), y.view(-1))
        return loss

    def compute_metric(self, y_hat, y):
        metric_fn = metric_factory(name=self.hparams.metric)
        try:  # if GPU metric
            metric = metric_fn(y_true=y, y_score=y_hat)
        except TypeError:  # if sklearn metric
            try:
                metric = metric_fn(
                    y_true=y.detach().cpu().numpy(),
                    y_score=y_hat.detach().cpu().numpy(),
                )
            except ValueError:
                metric = 0.50
        return metric

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat=y_hat.view(-1), y=y.view(-1))

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {
            "loss": loss,
            "y_hat": y_hat,
            "y": y,
        }

    def training_epoch_end(self, outputs: List):
        y_hat = torch.cat([out["y_hat"] for out in outputs], dim=0)
        y = torch.cat([out["y"] for out in outputs], dim=0)

        train_metric = self.compute_metric(y_hat=y_hat, y=y)
        self.log("train_metric", train_metric)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat=y_hat, y=y)
        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        return {"valid_loss": loss, "y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs: List):
        y_hat = torch.cat([out["y_hat"] for out in outputs], dim=0)
        y = torch.cat([out["y"] for out in outputs], dim=0)

        valid_metric = self.compute_metric(y_hat=y_hat, y=y)
        self.log("valid_metric", valid_metric)

        self.register_best_train_and_valid_metrics()
        self.print_metrics_to_console()

    def print_metrics_to_console(self):
        try:
            train_loss = self.trainer.callback_metrics["train_loss"]
            train_metric = self.trainer.callback_metrics["train_metric"]
            valid_loss = self.trainer.callback_metrics["valid_loss"]
            valid_metric = self.trainer.callback_metrics["valid_metric"]

            self.trainer.progress_bar_callback.main_progress_bar.write(
                f"Epoch {self.current_epoch} // train loss: {train_loss:.4f}, train metric: {train_metric:.4f}, valid loss: {valid_loss:.4f}, valid metric: {valid_metric:.4f}"
            )
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass

    def register_best_train_and_valid_metrics(self):
        # TODO: check if there is a better way to access this value
        try:
            train_metric = self.trainer.callback_metrics["train_metric"]
            valid_metric = self.trainer.callback_metrics["valid_metric"]
            if (
                self.best_valid_metric is None
                or valid_metric > self.best_valid_metric
            ):
                self.best_valid_metric = valid_metric
            if (
                self.best_train_metric is None
                or train_metric > self.best_train_metric
            ):
                self.best_train_metric = train_metric
        except (KeyError, AttributeError):
            # these errors occurs when in "tuning" mode (find optimal lr)
            pass