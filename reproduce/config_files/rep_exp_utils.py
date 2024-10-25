from typing import Callable, Any, Dict

import lightning as L
import torch
import torchmetrics
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch import nn

from cellm.components.cell_to_cell import CellToCellPytorchTransformer
from cellm.components.cell_to_output import CellToOutputMLP
from cellm.components.gene_to_cell import GeneToCellLinear
from cellm.components.masking import Masking
from cellm.data.data_structures import CellSample
import pickle
import scipy.stats
import pandas as pd

class CellClassifyModel(L.LightningModule):
    def __init__(self, num_genes, masking_strategy: Masking, lr: float = 1e-04, weight_decay: float = 0., dropout: float = 0., attn = 'nonlinear_attn'):
        super().__init__()

        # automatically access hparams with self.hparams.XXX
        self.save_hyperparameters(
            ignore=['gene_to_cell_encoder', 'cell_to_cell_encoder', 'cell_to_output_encoder', 'masking_strategy'])

        self.num_genes = num_genes
        self.masking_strategy = masking_strategy
        with open("/projects/site/gred/resbioai/liut61/disease_loader.pickle", 'rb') as handle:
            self.disease_label_dict = pickle.load(handle)
        # self.disease_number = len(self.disease_label_dict)
        self.disease_number = 2
        print(self.disease_number)

        self.gene_to_cell_encoder = GeneToCellLinear(self.num_genes, latent_dim=1024)
        # self.cell_to_cell_encoder = CellToCellPytorchTransformer(1024, n_heads=4, num_layers=2, single_cell_only=False)
        self.cell_to_output_encoder = CellToOutputMLP(input_dim=1024, output_dim=self.disease_number, hidden_dim=[512, 512], dropout=dropout)

        self.loss_func = self.get_loss_func()
        self.metrics = self.get_metrics()

        self.attn = attn

    def get_loss_func(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_metrics(self) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()
        metrics['accuracy'] = torchmetrics.Accuracy(task="multiclass", num_classes= self.disease_number)

        return metrics

    def compute_loss(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(preds, gt).mean()
        return loss

    def weighted_average(self, x):
        avg_atten = torch.nn.functional.softmax(x, dim=1)
        x_ = x * avg_atten 
        return x_.sum(axis=1)

    def forward(self,data, pad=None):
        if torch.cuda.is_available():
            x = data.cuda()
            if self.attn == 'transformer':
                pad = pad.cuda()
        else:
            x = data
        o = self.gene_to_cell_encoder(x)  # batch x sample x cell
        if self.attn == 'mean':
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'linear_attn':
            o_ = self.weighted_average(o)
        elif self.attn == 'nonlinear_attn':
            o_ = self.weighted_average_nonlinear(o)
        elif self.attn == 'transformer':
            o = self.cell_to_cell_encoder(o, src_key_padding_mask=~pad)
            o_ = o.mean(axis=1) # should be sample x embeddings
            
        o = self.cell_to_output_encoder(o_)  # sample x logits
        return o

    def compute_step(self, batch: CellSample, prefix: str, log=True) -> torch.Tensor:
        x = batch.x
        y_label = batch.disease_label
        o = self.gene_to_cell_encoder(x)  # batch x sample x cell
        # print(o.shape)
        # o = self.cell_to_cell_encoder(o, src_key_padding_mask=~batch.pad)  # batch x sample x cell
        # o = o.mean(axis=1) # should be sample x embeddings
        o = self.weighted_average(o)
        # print(o.shape) 
        o = self.cell_to_output_encoder(o)  # sample x logits
        
        preds = o
        gt = y_label # change it to y as labels

        loss = self.compute_loss(preds, gt)

        if log:
            self.log(f"{prefix}_loss", loss.item(), prog_bar=True, sync_dist=True)
            self._log_metric(prefix, preds, gt)
        return loss

    def _log_metric(self, prefix: str, logits: torch.Tensor, gt: torch.Tensor):
        for metric_name, metric_func in self.metrics.items():
            metric_str = f"{prefix}_{metric_name}"
            self.log(metric_str, metric_func(logits, gt).item(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def training_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='train')

    def validation_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='val')

    def test_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='test')

    def general_compute(self, batch: CellSample, batch_idx) -> torch.Tensor:
        if torch.cuda.is_available():
            x = batch.x.cuda()
        else:
            x = batch.x
        y_label = batch.disease_label
        o = self.gene_to_cell_encoder(x)  # batch x sample x cell
        # o = self.cell_to_cell_encoder(o, src_key_padding_mask=~batch.pad)  # batch x sample x cell
        # o = o.mean(axis=1) # should be batch x sample
        o = self.weighted_average(o)
        return o

    def obtain_embeddings_zs(self, x):
        o = self.gene_to_cell_encoder(x) 
        o = self.weighted_average(o)
        return o
        

    def obtain_annotation(self, batch:CellSample, batch_index):
        o = self.general_compute(batch, batch_index)
        o = self.cell_to_output_encoder(o)  # batch x sample x cell
        preds = o
        _, predicted = torch.max(preds, 1)
        probs = torch.softmax(o, 1)
        return predicted, probs

    def obtain_annotation_directly(self, x):
        o = self.forward(x)
        preds = o
        _, predicted = torch.max(preds, 1)
        probs = torch.softmax(o, 1)
        return predicted, probs

    def obtain_embeddings(self, batch:CellSample, batch_index) -> torch.Tensor:
        o = self.general_compute(batch, batch_index)
        return o # this o should be the patient embeddings
    

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