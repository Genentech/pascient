from logging import Logger

from jsonargparse import namespace_to_dict
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            # config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility

            dict_args = namespace_to_dict(self.config)

            # save number of model parameters
            dict_args["params"] = dict()
            dict_args["params"]["total"] = sum(p.numel() for p in pl_module.parameters())
            dict_args["params"]["trainable"] = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            dict_args["params"]["non_trainable"] = sum(p.numel() for p in pl_module.parameters() if not p.requires_grad)

            trainer.logger.log_hyperparams({"config": dict_args})
