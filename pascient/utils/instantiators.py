from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from pascient.utils import pylogger
import torchmetrics

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_metrics(metrics_cfg: DictConfig) -> List[torchmetrics.Metric]:
    """Instantiates metrics from config.

    :param metrics_cfg: A DictConfig object containing metric configurations.
    :return: A list of instantiated metrics.
    """
    metrics: List[torchmetrics.Metric] = []

    if not metrics_cfg:
        log.warning("No metric configs found! Skipping...")
        return metrics

    if not isinstance(metrics_cfg, DictConfig):
        raise TypeError("Metrics config must be a DictConfig!")

    for _, metric_conf in metrics_cfg.items():
        if isinstance(metric_conf, DictConfig) and "_target_" in metric_conf:
            log.info(f"Instantiating metric <{metric_conf._target_}>")
            metrics.append(hydra.utils.instantiate(metric_conf))

    return metrics

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger