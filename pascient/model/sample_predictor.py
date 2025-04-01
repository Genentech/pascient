from typing import Callable, Any, Dict

import lightning as L
import torch
import torchmetrics
from torch import nn

from pascient.components.masking import Masking
from pascient.data.data_structures import CellSample, SampleBatch
from pascient.model.patient_embedder import PatientEmbedder

from pascient.components.aggregators import Aggregator

from typing import Type, Union

import torch
from torch import nn

from typing import Union, List

import logging
log = logging.getLogger(__name__)


class SamplePredictor(PatientEmbedder):
    """ A simple model that takes in a gene expression matrix and predicts a label.

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
    sample_prediciton_loss:
        Whether to use the sample prediction loss.
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
                 patient_predictor: Union[nn.Module,None],
                 dropout: float = 0., 
                 metrics: List[torchmetrics.Metric] = [],
                 losses: Dict = None,
                 #cross_mask_loss: float = 0,
                 #cell_contrastive_loss: float = 0,
                 #sample_contrastive_loss: float = 0,
                 #sample_prediction_loss: float = 1,
                 compile: bool = False,
                 cell_contrastive_strategy: str = "classic"):

        super().__init__(num_genes = num_genes,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            masking_strategy = masking_strategy,
                            gene2cell_encoder = gene2cell_encoder,
                            cell2cell_encoder = cell2cell_encoder,
                            cell2patient_aggregation = cell2patient_aggregation,
                            cell2output = cell2output,
                            patient_encoder = patient_encoder,
                            cell_decoder = cell_decoder,
                            dropout = dropout,
                            metrics = metrics,
                            losses = losses,
                            compile = compile,
                            cell_contrastive_strategy = cell_contrastive_strategy)

        self.patient_predictor = patient_predictor

    def init_losses(self):
        if self.losses.sample_prediction_loss.weight > 0:
            self.sample_prediction_loss_func = self.losses.sample_prediction_loss.loss_fn()
            self.prediction_labels = self.losses.sample_prediction_loss.labels

    def compute_loss(self, 
                preds_cells: torch.Tensor,
                cell_embedding: Union[torch.Tensor,None], 
                p_embedding: torch.Tensor, 
                patient_preds: torch.Tensor,
                true_batch: SampleBatch) -> dict:
    
        losses = {}

        view_names = true_batch.view_names
        pad_mask = true_batch.padded_mask
        dropout_mask = true_batch.dropout_mask
         
        if self.losses.sample_prediction_loss.weight > 0:
            # sample prediction loss
            true_preds = [true_batch.sample_metadata[label] for label in self.prediction_labels]
            true_preds = torch.stack(true_preds,-1).to(patient_preds.dtype)
            losses["sample_prediction_loss"] = self.sample_prediction_loss_func(patient_preds, true_preds)
        return losses
    
    def compute_total_loss(self, losses):
        weights = {"sample_prediction_loss": self.losses.sample_prediction_loss.weight}
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

        patient_preds = self.patient_predictor(patient_embds)

        normalized_pat_embeds = torch.nn.functional.normalize(patient_embds, dim=-1)

        preds_cells = self.cell2output(cell_cross_embds)

        losses = self.compute_loss(preds_cells = preds_cells, 
                                cell_embedding = cell_cross_embds, 
                                p_embedding = normalized_pat_embeds,
                                patient_preds = patient_preds,
                                    true_batch = batch
                                )

        loss = self.compute_total_loss(losses)

        if log:
            self.log(f"{prefix}/loss", loss.item(), prog_bar=True, sync_dist=True, on_step= (prefix=="train"), on_epoch = True)
            for loss_name, loss_val in losses.items():
                self.log(f"{prefix}/{loss_name}", loss_val.item(), prog_bar=True, sync_dist=True, on_step= (prefix=="train"), on_epoch = True)
            
            self._update_metric( patient_embds = normalized_pat_embeds, 
                                preds_cells = preds_cells, 
                                patient_preds = patient_preds,
                                true_batch = batch)
            
        if prefix == "predict":
            return {"patient_preds": patient_preds, 
                    "patient_embds": normalized_pat_embeds,
                    "preds_cells": preds_cells,
                    "true_batch": batch}

        return loss
