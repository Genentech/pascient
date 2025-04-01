import abc
import time

import torch
from torch import optim
from lightning import LightningModule

from typing import Any, Dict, Tuple

class BaseLightningModule(LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.epoch_num = 0

    @abc.abstractmethod
    def get_loss(self, batch, prefix: str) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, _=None):
        start = time.time()
        loss = self.get_loss(batch, "train")
        delta = time.time() - start
        self.log("train/batch_time", delta, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _=None):
        return self.get_loss(batch, "val")

    def test_step(self, batch, _=None):
        return self.get_loss(batch, "test")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_train_epoch_end(self):
        self.epoch_num += 1
        print(f"Epoch: {self.epoch_num}", end="\r")

    def epoch_log(self, name: str, value, prefix: str):
        on_step = prefix == "train"
        self.log(f"{prefix}/{name}", value, on_epoch=True, on_step=on_step)
