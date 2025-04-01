from io import StringIO
import tqdm
import boto3
import pandas as pd

import torch
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
import hydra
import tqdm
import numpy as np

from captum.attr import IntegratedGradients

import argparse

# Load project root directory
import rootutils
rootutils.setup_root(search_from=".")

def load_model(config_path, checkpoint_path):
    # Run Name
    #run_name = "2025-02-09_22-25-15"
    #model_name = "epoch_002" 
    #config_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/.hydra/"
    #checkpoint_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/checkpoints/{model_name}.ckpt"

    with initialize_config_dir(version_base=None, config_dir=config_path, job_name="test_app"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True, 
                    overrides=["data.multiprocessing_context=null", "data.batch_size=16","data.sampler_cls._target_=cellm.data.data_samplers.BaseSampler","+data.output_map.return_index=True"])#, "data.num_workers=0","data.persistent_workers=False"])
        print(OmegaConf.to_yaml(cfg))

    checkpoint = torch.load(checkpoint_path)
    metrics = hydra.utils.instantiate(cfg.get("metrics"))
    model = hydra.utils.instantiate(cfg.model, metrics = metrics)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    datamodule = hydra.utils.instantiate(cfg.data)

    cfg.paths.output_dir = ""
    trainer = hydra.utils.instantiate(cfg.trainer)

    return model, datamodule

def load_binary_model(config_path, checkpoint_path):
    # Run Name
    #run_name = "2025-02-11_19-52-59"
    #model_name = "epoch_099" 
    #config_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/.hydra/"
    #checkpoint_path = f"/homefs/home/debroue1/projects/cellm/logs/train/runs/{run_name}/checkpoints/{model_name}.ckpt"

    with initialize_config_dir(version_base=None, config_dir=config_path, job_name="test_app"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True, 
                    overrides=["data.multiprocessing_context=null", "+data.output_map.return_index=True"])
        print(OmegaConf.to_yaml(cfg))

    checkpoint = torch.load(checkpoint_path)
    metrics = hydra.utils.instantiate(cfg.get("metrics"))
    model = hydra.utils.instantiate(cfg.model, metrics = metrics)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    datamodule = hydra.utils.instantiate(cfg.data)

    cfg.paths.output_dir = ""
    trainer = hydra.utils.instantiate(cfg.trainer)

    return model, datamodule