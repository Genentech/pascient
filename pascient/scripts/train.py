from typing import Any, Dict, List, Optional, Tuple

import lightning as L

from torchmetrics import Metric
import rootutils

from lightning import Callback, LightningDataModule, LightningModule, Trainer

import hydra

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from pascient.utils.pylogger import RankedLogger
from pascient.utils.instantiators import instantiate_callbacks, instantiate_loggers, instantiate_metrics
from pascient.utils.logging_utils import log_hyperparameters

log = RankedLogger(__name__, rank_zero_only = True)

from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
#from omegaconf import OmegaConf
#OmegaConf.register_new_resolver("eval", eval)



def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiate Metrics...>")
    metrics: List[Metric] = instantiate_metrics(cfg.get("metrics"))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, metrics = metrics)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    if len(logger) == 0:
        #check wether there is a callback that requires a logger
        callbacks = [cb for cb in callbacks if not isinstance(cb, L.pytorch.callbacks.LearningRateMonitor)]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, train_dataloaders = datamodule.train_dataloader(), val_dataloaders = datamodule.val_dataloader(), ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    ## merge train and test metrics
    #metric_dict = {**train_metrics, **test_metrics}

    #return metric_dict, object_dict

    return



@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    #extras(cfg)

    # train the model
    #metric_dict, _ = 
    train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    #metric_value = get_metric_value(
    #    metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    #)

    # return optimized metric
    #return metric_value

    return


if __name__ == "__main__":
    main()