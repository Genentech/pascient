import logging
import time

from lightning.pytorch.cli import LightningCLI

from cellm.data.data_scimilarity import SampleCellsDataModuleCustom
from cellm.model.cell_masking import CellMaskingModel,CellClassifyModel
from cellm.utils.cli_utils import LoggerSaveConfigCallback

logging.getLogger().setLevel(logging.INFO)

import random
from lightning.pytorch import Trainer, seed_everything
import multiprocessing as mp
import os
os.environ["WANDB_CACHE_DIR"] = "/home/liut61/scratch/wandb/"



class CellmaskingLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        if 'fit' in self.config.keys():
            time_name = time.strftime("%Y%m%d-%H%M%S")
            previous_name = self.config['fit']['trainer']['logger']['init_args']['name']
            if previous_name is None:
                self.config['fit']['trainer']['logger']['init_args']['name'] = time_name
            else:
                self.config['fit']['trainer']['logger']['init_args']['name'] = time_name + '_' + previous_name


def main():
    
    # note that mp apparently re-executes the whole file, so calling `set_start_method`
    # without the check here causes errors.
    if mp.get_start_method() is None:
        mp.set_start_method('spawn')
    seed_everything(0, workers=True)
    cli = CellmaskingLightningCLI(model_class=CellClassifyModel, datamodule_class=SampleCellsDataModuleCustom, 
                                  save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":

    import wandb
    wandb.init(project="diseaseclass")
    main()
