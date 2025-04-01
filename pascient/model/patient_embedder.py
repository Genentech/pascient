from typing import Callable, Any, Dict

import lightning as L
import torch
import torchmetrics
from torch import nn

from pascient.components.masking import Masking
from pascient.data.data_structures import CellSample, SampleBatch

from pascient.components.aggregators import Aggregator

from typing import Type, Union

import torch
from torch import nn

from typing import Union, List

import logging
log = logging.getLogger(__name__)


class PatientEmbedder(L.LightningModule):
    """ A simple model that takes in a gene expression matrix and predicts the cell embeddings.

    Parameters
    ----------
    num_genes:
        Number of genes in the input data.
    optimizer:
        Optimizer to use.
    scheduler:
        Scheduler to use.
    masking_strategy:
        Masking strategy to use.
    gene2cell_encoder: 
        Gene to cell encoder.
    cell2cell_encoder:
        Cell to cell encoder.
    cell2patient_aggregation:
        Aggregator to use to aggregate the cell embeddings to patient embeddings.
    patient_encoder:
        Patient encoder.
    cell_decoder:
        Cell decoder.
    dropout:
        Dropout to use.
    cross_mask_loss:
        Whether to use the cross mask loss.
    contrastive_loss:
        Whether to use the contrastive loss.
    sample_contrastive_loss:
        Whether to use the sample contrastive loss.
    patient_embedding_strategy:
        Strategy to use to aggregate the cell embeddings to patient embeddings.
    compile:
        Whether to compile the model. (torch 2.0)
    
    """
    def __init__(self, num_genes, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler, 
                 masking_strategy: Masking, 
                 gene2cell_encoder: nn.Module,
                 cell2cell_encoder: Union[nn.Module,None],
                 cell2patient_aggregation: Aggregator,
                 cell2output: Union[nn.Module,None],
                 patient_encoder: Union[nn.Module,None],
                 cell_decoder: Union[nn.Module,None],
                 dropout: float = 0., 
                 metrics: List[torchmetrics.Metric] = [],
                 losses: Dict = None, 
                 compile: bool = False,
                 cell_contrastive_strategy: str = "ancestry"):
        
        super().__init__()

        # automatically access hparams with self.hparams.XXX
        self.save_hyperparameters(
            ignore=['masking_strategy'])

        self.num_genes = num_genes
        self.masking_strategy = masking_strategy

        self.gene2cell_encoder = gene2cell_encoder
        self.cell_decoder = cell_decoder
        self.cell2patient_aggregation = cell2patient_aggregation
        self.patient_encoder = patient_encoder
        self.cell2cell_encoder = cell2cell_encoder
        self.cell2output = cell2output

        self.metrics = metrics

        # what losses to consider here
        self.losses = losses

        self.cell_contrastive_strategy = cell_contrastive_strategy
        log.info(f"Cell contrastive strategy: {self.cell_contrastive_strategy}")

        self.init_losses()

    def init_losses(self):
        if self.losses.cross_mask_loss.weight > 0:
            self.cross_mask_loss_func = self.losses.cross_mask_loss.loss_fn()
        if self.losses.cell_contrastive_loss.weight > 0:
            self.cell_contrastive_loss_func = self.losses.cell_contrastive_loss.loss_fn()
        if self.losses.sample_contrastive_loss.weight > 0:
            self.sample_contrastive_loss_func = self.losses.sample_contrastive_loss.loss_fn()
 
    def get_cross_mask_loss_func(self) -> nn.Module:
        return self.cross_mask_loss_func

    def get_contrastive_loss_func(self) -> nn.Module:
        return self.contrastive_loss_func


    def compute_loss(self, 
                    preds_cells: torch.Tensor,
                    cell_embedding: Union[torch.Tensor,None], 
                    p_embedding: torch.Tensor, 
                    true_batch: SampleBatch) -> dict:
        
        losses = {}

        view_names = true_batch.view_names
        pad_mask = true_batch.padded_mask
        dropout_mask = true_batch.dropout_mask
        
        if self.losses.cross_mask_loss.weight > 0:
            
            true_x, pred_x = self.masking_strategy.compare_reconstructions(true_batch, preds_cells)

            # select only the views we want to consider.
            #gt_views_idx = [view_names.index(view) for view in self.masking_strategy.views] # for the ground truth
            #rec_views_idx = [view_names.index(f"mask_from_{view}") for view in self.masking_strategy.views] # for the reconstruction
            
            # select only the cells that are non padded and masked.
            #loss_mask = pad_mask[:,gt_views_idx][...,None] * ~(dropout_mask.bool())

            #true_x = true_batch.x[:,gt_views_idx][loss_mask]
            #pred_x = preds_cells[:,rec_views_idx][loss_mask]

            losses["cross_mask_loss"] = self.cross_mask_loss_func(pred_x, true_x).mean()

        if self.losses.cell_contrastive_loss.weight > 0:

            self.cell_contrastive_views = ["view_0", "view_1"] # which views are taken into account for the contrastive loss.
            self.cell_contrastive_label = "celltype_id" # which label is used for the contrastive loss.

            view_idxs = [view_names.index(view) for view in self.cell_contrastive_views]

            normalized_cell_embeddings = torch.nn.functional.normalize(cell_embedding[:,view_idxs][pad_mask[:,view_idxs]], dim=-1)[:,None] # need extra dimension for the repetition dimension. 
            
            labels = true_batch.cell_metadata.cell_level_labels[self.cell_contrastive_label][:,view_idxs][pad_mask[:,view_idxs]]

            if self.cell_contrastive_strategy == "ancestry": # use the ancestry matrix to compute the contrastive loss - removing the ancestors and childs from the denominator
                ontology_matrix = true_batch.cell_metadata.ancestry_matrix
            elif self.cell_contrastive_label == "classic": # only uses the labels and ignores the ancestry matrix
                ontology_matrix = None

            losses["cell_contrastive_loss"] = self.cell_contrastive_loss_func(features = normalized_cell_embeddings, labels = labels, ontology_matrix = ontology_matrix)

        if self.losses.sample_contrastive_loss.weight > 0:
            # contrastive loss at the patient level

            losses["sample_contrastive_loss"] = self.sample_contrastive_loss_func(p_embedding)
            
        return losses

    def compute_total_loss(self, losses):
        weights = {"sample_contrastive_loss": self.sample_contrastive_loss,
                   "cross_mask_loss": self.cross_mask_loss,
                   "cell_contrastive_loss": self.cell_contrastive_loss}
        loss = 0
        for k, v in losses.items():
            loss += weights[k] * v
        return loss

    def compute_step(self, batch: SampleBatch, prefix: str, log=True) -> torch.Tensor:
        

        batch = self.masking_strategy(batch)
        x_full = batch.x
        padding_mask = batch.padded_mask

        cell_embds = self.gene2cell_encoder(x_full)

        cell_cross_embds = self.cell2cell_encoder(cell_embds, padding_mask = padding_mask)

        patient_embds = self.cell2patient_aggregation.aggregate(data = cell_cross_embds, mask = padding_mask)

        patient_embds = self.patient_encoder(patient_embds)

        normalized_pat_embeds = torch.nn.functional.normalize(patient_embds, dim=-1)

        preds_cells = self.cell2output(cell_cross_embds)
 
        losses = self.compute_loss(preds_cells = preds_cells, 
                                 cell_embedding = cell_cross_embds, 
                                 p_embedding = normalized_pat_embeds,
                                    true_batch = batch
                                )

        loss = self.compute_total_loss(losses)

        if log:
            self.log(f"{prefix}/loss", loss.item(), prog_bar=True, sync_dist=True, on_step= (prefix=="train"), on_epoch = True)
            for loss_name, loss_val in losses.items():
                self.log(f"{prefix}/{loss_name}", loss_val.item(), prog_bar=True, sync_dist=True, on_step= (prefix=="train"), on_epoch = True)
            
            self._update_metric( patient_embds = normalized_pat_embeds, 
                                preds_cells = preds_cells, 
                                true_batch = batch)

        return loss

    def _update_metric(self, **kwargs):
        for metric in self.metrics:
            #Compute the metrics and log them
            metric.update(masking_strategy = self.masking_strategy, **kwargs)

    def training_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='train')

    def validation_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='val')
    
    def _compute_metrics(self, prefix):
        # Compute the aggregated metrics and log them
        for metric in self.metrics:
            metric_name = metric.name
            metric_str = f"{prefix}/{metric_name}"
            self.log(metric_str, metric.compute(), prog_bar = True, sync_dist=True, on_step = False, on_epoch = True)

    def on_training_epoch_end(self):
        # Compute the aggregated metrics and log them
        self._compute_metrics("train")
    
    def on_validation_epoch_end(self):
        # Compute the aggregated metrics and log them
        self._compute_metrics("val")

    def on_test_epoch_end(self):
        # Compute the aggregated metrics and log them
        self._compute_metrics("test")
    
    def on_fit_start(self):
        # fix metrics devices
        for metrics in self.metrics:
            metrics.to(self.device)

    def on_test_start(self):
        # fix metrics devices
        for k, v in self.metrics.items():
            self.metrics[k] = v.to(self.device)

    def predict_step(self, batch: CellSample, batch_idx):
        return self.compute_step(batch, prefix='predict', log = False)
     
    def test_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='test')

    def configure_optimizers(self) -> Dict[str, Any]:

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
