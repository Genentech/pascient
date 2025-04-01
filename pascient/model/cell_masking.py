from typing import Callable, Any, Dict

import lightning as L
import torch
import torchmetrics
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch import nn

from pascient.components.cell_to_cell import CellToCellPytorchTransformer
from pascient.components.cell_to_output import CellToOutputMLP
from pascient.components.gene_to_cell import GeneToCellLinear
from pascient.components.masking import Masking
from pascient.data.data_structures import CellSample


class CellMaskingModel(L.LightningModule):
    def __init__(self, num_genes, masking_strategy: Masking, lr: float = 1e-04, weight_decay: float = 0.):
        super().__init__()

        # automatically access hparams with self.hparams.XXX
        self.save_hyperparameters(
            ignore=['gene_to_cell_encoder', 'cell_to_cell_encoder', 'cell_to_output_encoder', 'masking_strategy'])

        self.num_genes = num_genes
        self.masking_strategy = masking_strategy

        self.gene_to_cell_encoder = GeneToCellLinear(self.num_genes, latent_dim=32)
        self.cell_to_cell_encoder = CellToCellPytorchTransformer(32, n_heads=4, num_layers=2, single_cell_only=False)
        self.cell_to_output_encoder = CellToOutputMLP(input_dim=32, output_dim=self.num_genes, hidden_dim=[16, 16])

        self.loss_func = self.get_loss_func()
        self.metrics = self.get_metrics()

    def get_loss_func(self) -> nn.Module:
        return nn.MSELoss(reduction='none')

    def get_metrics(self) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()
        metrics['r2_cell'] = torchmetrics.R2Score(num_outputs=self.gene_to_cell_encoder.n_genes)

        return metrics

    def compute_loss(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(preds, gt).mean()
        return loss

    def compute_step(self, batch: CellSample, prefix: str, log=True) -> torch.Tensor:
        
        x = batch.x
        if hasattr(batch, 'mask'):  # precomputed mask
            mask = batch.mask
            x_masked, mask = self.masking_strategy.apply_mask(x, mask)
        else:
            x_masked, mask = self.masking_strategy(x)

        o = self.gene_to_cell_encoder(x_masked)  # batch x sample x cell
        o = self.cell_to_cell_encoder(o, src_key_padding_mask=~batch.pad)  # batch x sample x cell
        o = self.cell_to_output_encoder(o)  # batch x sample x cell

        # loss: keep only masked genes (mask=False), and not padded cells (pad=True)
        loss_mask = (~mask) & batch.pad[..., None]

        preds_masked = o[loss_mask]
        gt_masked = x[loss_mask]

        loss = self.compute_loss(preds_masked, gt_masked)

        if log:
            self.log(f"{prefix}/loss", loss.item(), prog_bar=True, sync_dist=True)
            self._log_metric(prefix, preds_masked, gt_masked)
        return loss

    def _log_metric(self, prefix: str, logits: torch.Tensor, gt: torch.Tensor):
        for metric_name, metric_func in self.metrics.items():
            metric_str = f"{prefix}/{metric_name}"
            self.log(metric_str, metric_func(logits, gt).item(), prog_bar=True, sync_dist=True)

    def training_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='train')

    def validation_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='val')

    def test_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='test')

    def on_fit_start(self):
        # fix metrics devices
        for k, v in self.metrics.items():
            self.metrics[k] = v.to(self.device)

    def on_test_start(self):
        # fix metrics devices
        for k, v in self.metrics.items():
            self.metrics[k] = v.to(self.device)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
