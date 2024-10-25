import logging
import time

from lightning.pytorch.cli import LightningCLI

from cellm.data.data_scimilarity import SampleCellsDataModuleCustom
from cellm.model.cell_masking import CellMaskingModel,CellClassifyModel
from cellm.utils.cli_utils import LoggerSaveConfigCallback

logging.getLogger().setLevel(logging.INFO)


class CellmaskingLightningCLI(LightningCLI):
    def before_instantiate_classes(self):
        time_name = time.strftime("%Y%m%d-%H%M%S")
        previous_name = self.config['fit']['trainer']['logger']['init_args']['name']
        if previous_name is None:
            self.config['fit']['trainer']['logger']['init_args']['name'] = time_name
        else:
            self.config['fit']['trainer']['logger']['init_args']['name'] = time_name + '_' + previous_name


def main():
    cli = CellmaskingLightningCLI(model_class=CellMaskingModel, datamodule_class=SampleCellsDataModuleCustom,
                                  save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":

    import wandb
    wandb.init(project="celllmnew")
    main()
