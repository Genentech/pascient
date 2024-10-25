from typing import Callable, Any, Dict

import lightning as L
import torch
import torchmetrics
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch import nn

from cellm.components.cell_to_cell import CellToCellPytorchTransformer
from cellm.components.cell_to_output import CellToOutputMLP
from cellm.components.gene_to_cell import GeneToCellLinear,GeneToCellMLP
from cellm.components.masking import Masking
from cellm.data.data_structures import CellSample
import pickle

from pytorch_metric_learning.losses import SelfSupervisedLoss, TripletMarginLoss

class MultiTaskLoss(torch.nn.Module):
  '''https://arxiv.org/abs/1705.07115'''
  def __init__(self, is_regression, reduction='none'):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses

class CellClassifyModel(L.LightningModule):
    def __init__(self, num_genes, masking_strategy: Masking, lr: float = 1e-04, weight_decay: float = 0., dropout: float = 0., text_emb = False, attn = 'linear_attn', n_dim = 1024, classify_mode = 'binary', num_layers=2, label_smoothing=0.0, logit_adjustment=False, residual=False, batchnorm = False, include_tissue=False, uncertainty=False, noise_robust=False, label_weight = False, mask_training = False, decoupling = False, n_hidden = 512, n_hidden_num = 2, contras=False, contras_ct = False, gmlp_seqlen = 100):
        super().__init__()

        # automatically access hparams with self.hparams.XXX
        self.save_hyperparameters(
            ignore=['gene_to_cell_encoder', 'cell_to_cell_encoder', 'cell_to_output_encoder', 'masking_strategy'])

        self.num_genes = num_genes
        self.masking_strategy = masking_strategy
        if (classify_mode == 'binary') or (classify_mode == 'binary_ct'):
            with open("/projects/site/gred/resbioai/liut61/disease_loader_2label.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        elif classify_mode == 'multilabel':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap9.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        elif classify_mode == 'multilabel_final':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_all.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
            # self.disease_number = len(self.disease_label_dict) - 1
        self.disease_number = len(self.disease_label_dict)
        print(self.disease_number)
        self.classify_mode = classify_mode

        with open("/projects/site/gred/resbioai/liut61/tissue_loader_all.pickle", 'rb') as handle:
            self.tissue_label_dict = pickle.load(handle)
            self.tissue_number = len(self.tissue_label_dict)

        if include_tissue == 'clean':
            with open("/projects/site/gred/resbioai/liut61/tissue_clean_loader.pickle", 'rb') as handle:
                self.tissue_label_dict = pickle.load(handle)
                self.tissue_number = len(self.tissue_label_dict)

        self.uncertainty = uncertainty 
        if self.uncertainty:
            self.uw_class = MultiTaskLoss(torch.Tensor([False, False]), 'mean')
        # self.label_key = {}
        # for key,value in self.disease_label_dict.items():
        #     self.label_key[value] = key
            
        # with open("/projects/site/gred/resbioai/liut61/disease_gpt4_description_embeddings.pickle", 'rb') as handle:
        #     self.disease_emb_dict = pickle.load(handle)
        # self.disease_number = 2
        print(self.disease_number)
        self.logit_adjustment = logit_adjustment
        self.include_tissue = include_tissue
        self.label_weight = label_weight

        if residual:
                self.gene_to_cell_encoder = GeneToCellMLP(self.num_genes, latent_dim=n_dim, hidden_dim = [n_hidden] * n_hidden_num, residual=residual, batchnorm=batchnorm, dropout = dropout)
        else:
                self.gene_to_cell_encoder = GeneToCellLinear(self.num_genes, latent_dim=n_dim)
            

        if self.include_tissue != False:
            self.cell_to_output_tissue_encoder = CellToOutputMLP(input_dim=n_dim , output_dim=self.tissue_number, hidden_dim=[n_hidden] * n_hidden_num, dropout=dropout,residual=residual, batchnorm=batchnorm) # can be 512, 512
        
        if attn  == 'transformer':
            self.cell_to_cell_encoder = CellToCellPytorchTransformer(n_dim , n_heads=4, num_layers=num_layers, single_cell_only=False)

        if attn =='gmlp':
            self.cell_to_cell_encoder = gMLP(
                                num_tokens = None,
                                dim = n_dim,
                                depth = 1,
                                seq_len = gmlp_seqlen,
                                circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
                                act = nn.PReLU()              # activation for spatial gate (defaults to identity)
                            )

        if attn == 'gated attention':
            self.attention_V = nn.Sequential(
                nn.Linear(n_dim, n_dim),
                nn.Tanh(),
            )

            self.attention_U = nn.Sequential(
                nn.Linear(n_dim, n_dim),
                nn.Sigmoid(),
            )
            self.attention_weights = nn.Linear(n_dim, 1, bias=False)
            
        self.cell_to_output_encoder = CellToOutputMLP(input_dim=n_dim , output_dim=self.disease_number, hidden_dim=[n_hidden] * n_hidden_num, dropout=dropout,residual=residual, batchnorm=batchnorm) # can be 512, 512

        if logit_adjustment:
            if self.disease_number == 10:
                self.logit_adj = torch.load("/projects/site/gred/resbioai/liut61/adjusted_logits_data.pkl")
                logit_adj_list = list(self.logit_adj)
                logit_adj_list.insert(4, -10) # hard code to exclude class 4, which is None.
                self.logit_adj = torch.FloatTensor(logit_adj_list)
                print(self.logit_adj)
            else:
                self.logit_adj = torch.load("/projects/site/gred/resbioai/liut61/adjusted_logits_data.pkl")
                print(self.logit_adj)
            
        # if text_emb:
        #     self.cell_to_output_encoder = CellToOutputMLP(input_dim=1024 + 1024, output_dim=self.disease_number, hidden_dim=[512, 512], dropout=dropout)
        self.attn = attn
        self.text_emb = text_emb
        if text_emb:
            self.loss_ssf = SelfSupervisedLoss(TripletMarginLoss())
        self.contras = False
        if contras:
            self.contras = contras
            self.loss_ssf_con =  NTXentLoss()
        self.contras_ct = False
        if contras_ct:
            self.contras_ct = contras_ct
            self.loss_ssf_con =  NTXentLoss()
            
        if attn == 'nonlinear_attn':
            self.attention = nn.Sequential(
                nn.Linear(n_dim , n_dim ), # matrix V
                nn.Tanh(),
                # nn.PReLU(),
                nn.Linear(n_dim, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
            )

        self.noise_robust = noise_robust
        self.mask_training = mask_training

        if self.mask_training:
            self.mask_loss = nn.MSELoss(reduction='none')
            self.cell_to_rec = CellToOutputMLP(input_dim=n_dim, output_dim=self.num_genes, hidden_dim=[n_hidden] * n_hidden_num)

        self.decoupling = decoupling
        if self.decoupling:
            for param in self.gene_to_cell_encoder.parameters():
                param.requires_grad = False
            # if attn  == 'transformer':
            #     for param in self.cell_to_cell_encoder.parameters():
            #         param.requires_grad = False
            # if attn == 'nonlinear_attn':
            #     for param in self.attention.parameters():
            #         param.requires_grad = False
            self.cell_to_output_encoder = CellToOutputMLP(input_dim=n_dim , output_dim=self.disease_number, hidden_dim=[n_hidden] * n_hidden_num, dropout=dropout,residual=residual, batchnorm=batchnorm) # can be 512, 512, init this module

        self.loss_func = self.get_loss_func(label_smoothing = label_smoothing)
        self.metrics = self.get_metrics()
        if self.decoupling:
            checkpoint = torch.load("/home/liut61/scratch/disease_class/disease2classlr1e4_wd1e-4_batch64_epoch20_drop0.0_cells1000_mutlilabel_nodecoupling_nonlinear_attn_epoch=17-val_accuracy=0.73.ckpt")
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            
            

    def get_loss_func(self, label_smoothing) -> nn.Module:
  #       return nn.CrossEntropyLoss(weight = torch.tensor([ 0.024203821656050957, 0.0021231422505307855,   0.10912951167728238,
  #  0.02505307855626327,    0.1851380042462845,     0.608067940552017,
  #  0.02038216560509554,  0.008492569002123142,  0.004670912951167728,
  # 0.012738853503184714]))
        if self.label_weight == False:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            if self.classify_mode == 'binary':
                weight = [1370, 245]
                return nn.CrossEntropyLoss(weight = 1.0 / torch.tensor(weight) , label_smoothing=label_smoothing)
            
            if self.decoupling and self.label_weight:
                weight = [57,5, 268, 59, 1443, 48, 20, 11, 30]
                return nn.CrossEntropyLoss(weight = 1.0 / torch.tensor(weight) , label_smoothing=label_smoothing)
            if self.classify_mode == 'multilabel_final':
                with open("/projects/site/gred/resbioai/liut61/disease_weight_all.pickle", 'rb') as handle:
                    weight = pickle.load(handle)
                return nn.CrossEntropyLoss(weight = 1.0 / (torch.tensor(weight) +1e-8) , label_smoothing=label_smoothing)
            else:
                weight = [57,5, 268, 59, 1443, 48, 20, 11, 30]
                return nn.CrossEntropyLoss(weight = 1.0 / torch.tensor(weight) , label_smoothing=label_smoothing)
    

    def get_metrics(self) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()
        # metrics['accuracy'] = torchmetrics.Accuracy(task="multiclass", num_classes= self.disease_number, average='weighted')
        metrics['accuracy'] = torchmetrics.Accuracy(task="multiclass", num_classes= self.disease_number)

        return metrics

    def compute_loss(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(preds, gt).mean()
        return loss

    def weighted_average_nonlinear(self, x):
        # avg_atten = torch.nn.functional.softmax(x, dim=1)
        # x_ = x * avg_atten 
        attn = self.attention(x)
        attn = torch.nn.functional.softmax(attn, dim=1)
        x_ = attn * x
        return x_.sum(axis=1)

    def visualize_attention(self, x):
        attn = self.attention(x)
        attn = torch.nn.functional.softmax(attn, dim=1)
        return attn[:,:,0]
        

    def weighted_average(self, x):
        avg_atten = torch.nn.functional.softmax(x, dim=1)
        x_ = x * avg_atten 
        return x_.sum(axis=1)

    def compute_step(self, batch: CellSample, prefix: str, log=True) -> torch.Tensor:
        if self.decoupling:
            self.loss_func = nn.CrossEntropyLoss()
        x = batch.x
        y_label = batch.disease_label
        if (prefix == 'train') and self.noise_robust:
            x = torch.cat((x, x + torch.randn(x.shape).to(self.device)), 0)
            y_label = torch.cat((y_label,y_label), 0)
            
        if self.mask_training:
            if hasattr(batch, 'mask'):  # precomputed mask
                mask = batch.mask
                x_masked, mask = self.masking_strategy.apply_mask(x, mask)
            else:
                x_masked, mask = self.masking_strategy(x)
        
            o = self.gene_to_cell_encoder(x_masked)  # batch x sample x cell
        else:
            o = self.gene_to_cell_encoder(x)  # batch x sample x cell
        # print(o.shape)
        # o = self.cell_to_cell_encoder(o, src_key_padding_mask=~batch.pad)  # batch x sample x cell
        # o = o.mean(axis=1) # should be sample x embeddings
    

        if self.attn == 'mean':
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'linear_attn':
            o_ = self.weighted_average(o)
        elif self.attn == 'nonlinear_attn':
            o_ = self.weighted_average_nonlinear(o)
        elif self.attn == 'transformer':
            o = self.cell_to_cell_encoder(o, src_key_padding_mask=~batch.pad)
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'gmlp':
            o = self.cell_to_cell_encoder(o)
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'gated attention':
            A_V = self.attention_V(o)  # NxD
            A_U = self.attention_U(o)  # NxD
            A = self.attention_weights(A_V * A_U)  # element wise multiplication # Nx1
            A = torch.nn.functional.softmax(A.transpose(-1, -2), dim=-1)
            o_ = torch.bmm(A, o).squeeze(dim=1)
            # A = torch.nn.functional.softmax(A.transpose(1,0), dim=1)
            # o_ = torch.bmm(A, o).squeeze(dim=1)
            
        preds = self.cell_to_output_encoder(o_)  # sample x logits

        if self.logit_adjustment:
            preds = preds + self.logit_adj.to(o.device)
        gt = y_label # change it to y as labels
        loss = self.compute_loss(preds, gt)

        if self.contras_ct:
            for i in range(len(o)):
                o_filter = o[i,:,:]
                loss += self.contras_ct * self.loss_ssf_con(o_filter, batch.celltype_label[i])

        if self.contras:
            loss += self.contras * self.loss_ssf_con(o_, gt)

        if self.mask_training:
            o = self.cell_to_rec(o)
            # loss: keep only masked genes (mask=False), and not padded cells (pad=True)
            loss_mask = (~mask) & batch.pad[..., None]
    
            preds_masked = o[loss_mask]
            gt_masked = x[loss_mask]
    
            loss += self.mask_loss(preds_masked, gt_masked).mean()

        # text embeddings similarity learning
        if self.text_emb:
            # loss += torch.nn.functional.mse_loss(o_, batch.disease_emb)
            loss += self.loss_ssf(o_, batch.disease_emb)

        if self.include_tissue != False:
            o_t = self.cell_to_output_tissue_encoder(o_)
            if self.uncertainty:
                # loss = self.uw_class(torch.cat((loss.reshape(1), self.compute_loss(o_t, batch.tissue_label))), dim=0)
                loss = self.uw_class(torch.stack((loss, self.compute_loss(o_t, batch.tissue_label))))
            else:
                loss += self.compute_loss(o_t, batch.tissue_label)

        if log:
            self.log(f"{prefix}_loss", loss.item(), prog_bar=True, sync_dist=True)
            # self._log_metric(prefix, preds, gt)

        if prefix == 'val':
            self._log_metric(prefix, preds, gt)
    
        return loss

    def _log_metric(self, prefix: str, logits: torch.Tensor, gt: torch.Tensor):
        for metric_name, metric_func in self.metrics.items():
            metric_str = f"{prefix}_{metric_name}"
            # self.log(metric_str, metric_func(logits, gt).item(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
            # self.log(metric_str, metric_func(logits, gt).item(), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            metric_func.update(logits, gt)

    def training_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='train')

    def validation_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='val')
        
    def on_validation_epoch_end(self):
        for metric_name, metric_func in self.metrics.items():
            metric_str = f"val_{metric_name}"
            self.log(metric_str, metric_func.compute(), prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
            metric_func.reset()

    def test_step(self, batch: CellSample, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix='test')

    def general_compute(self, batch: CellSample, batch_idx) -> torch.Tensor:
        if torch.cuda.is_available():
            x = batch.x.cuda()
            pad = batch.pad.cuda()
        else:
            x = batch.x
            pad = batch.pad
        o = self.gene_to_cell_encoder(x)
        y_label = batch.disease_label
        if self.attn == 'mean':
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'linear_attn':
            o_ = self.weighted_average(o)
        elif self.attn == 'nonlinear_attn':
            o_ = self.weighted_average_nonlinear(o)
        elif self.attn == 'transformer':
            o = self.cell_to_cell_encoder(o, src_key_padding_mask=~pad)
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'gmlp':
            o = self.cell_to_cell_encoder(o)
            o_ = o.mean(axis=1) # should be sample x embeddings
        elif self.attn == 'gated attention':
            A_V = self.attention_V(o)  # Nxdim
            A_U = self.attention_U(o)  # Nxdim
            A = self.attention_weights(A_V * A_U)  
            A = torch.transpose(A, -1, -2)  # 1xN
            A = F.softmax(A, dim=-1)  # softmax over N
            o_ = torch.bmm(A, o).squeeze(dim=1)
        return o_
        

    def obtain_annotation(self, batch:CellSample, batch_index):
        o = self.general_compute(batch, batch_index)
        # if self.text_emb:
        #     o = torch.cat([o, batch.disease_emb], axis=1)
        o = self.cell_to_output_encoder(o)  # batch x sample x cell

        if self.logit_adjustment:
            o = o + self.logit_adj.to(o.device)
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer
        # return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val_loss",
        #             "interval": "epoch",
        #             # If "monitor" references validation metrics, then "frequency" should be set to a
        #             # multiple of "trainer.check_val_every_n_epoch".
        #         },
        #     }


import os
from typing import List, Optional

import torch
from lightning import LightningDataModule
from cellm.data.data_scimilarity_gred import SampleCellsDataModule_disease, scDataset_disease
from torch.utils.data import Dataset, DataLoader

# from cellm.data.data_structures import CellSample
import pickle

from dataclasses import dataclass

import torch
import json


@dataclass
class CellSample:
    x: torch.Tensor
    pad: torch.Tensor
    disease_label: torch.Tensor
    disease_emb: torch.Tensor
    tissue_label: torch.Tensor

    @staticmethod
    def collate(batch):
        device = batch[0].x.device
        
        # Initialize a dictionary to store collated tensors
        collated = {
            key: torch.stack([getattr(b, key) for b in batch], dim=0).to(device=device)
            for key in batch[0].__dict__.keys()
        }
        
        return CellSample(**collated)

@dataclass
class CellDataMasked(CellSample):
    mask: torch.Tensor


tiledb_base_path = '/projects/global/gred/resbioai/CeLLM/tiledb'

CELLURI = "scimilarity_human_10x_cell_metadata"
GENEURI = "scimilarity_human_10x_gene_metadata"
COUNTSURI = "scimilarity_human_10x_counts"

cell_tdb_uri = os.path.join(tiledb_base_path, CELLURI)
gene_tdb_uri = os.path.join(tiledb_base_path, GENEURI)
counts_tdb_uri = os.path.join(tiledb_base_path, COUNTSURI)

gene_order = "/gstore/data/omni/scdb/cleaned_h5ads/gene_order.tsv"


# def get_default_val_studies():
#     val_studies = [
#         "24d42e5e-ce6d-45ff-a66b-a3b3b715deaf",
#         "29f92179-ca10-4309-a32b-d383d80347c1",
#         "2a79d190-a41e-4408-88c8-ac5c4d03c0fc",
#         "60358420-6055-411d-ba4f-e8ac80682a2e",
#         "7d7cabfd-1d1f-40af-96b7-26a0825a306d",
#         "8191c283-0816-424b-9b61-c3e1d6258a77",
#         "9b02383a-9358-4f0f-9795-a891ec523bcc",
#         "a98b828a-622a-483a-80e0-15703678befd",
#         "b3e2c6e3-9b05-4da9-8f42-da38a664b45b",
#         "be21c2d1-2392-47d0-96fb-c625d115e0dc",
#         "DS000010060",
#         "DS000010475",
#         "DS000011735",
#         "e2a4a67f-6a18-431a-ab9c-6e77dd31cc80",
#         "fcb3d1c1-03d2-41ac-8229-458e072b7a1c",
#     ]
#     return val_studies

def get_default_val_studies(classify_mode='binary'):
 #    val_studies = ['9b02383a-9358-4f0f-9795-a891ec523bcc',
 # 'fcb3d1c1-03d2-41ac-8229-458e072b7a1c',
 # 'a98b828a-622a-483a-80e0-15703678befd',
 # '29f92179-ca10-4309-a32b-d383d80347c1',
 # '8191c283-0816-424b-9b61-c3e1d6258a77',
 # 'b3e2c6e3-9b05-4da9-8f42-da38a664b45b',
 # 'DS000010060',
 # 'DS000011735',
 # '24d42e5e-ce6d-45ff-a66b-a3b3b715deaf',
 # 'DS000010475']

    if classify_mode == 'binary':
        val_studies = ['ddfad306-714d-4cc0-9985-d9072820c530',
                              'DS000010042',
     '436154da-bcf1-4130-9c8b-120ff9a888f2',
                              'DS000010028',
     '7d7cabfd-1d1f-40af-96b7-26a0825a306d',
                                'GSE158034',
                                'GSE158030',
                                'GSE159113',
                                'GSE152522',
                                'GSE153421',
                                'GSE139324',
                                'GSE144430',
                                'GSE134004',
                                'GSE128879']
    elif classify_mode == 'multilabel':
        with open('/projects/site/gred/resbioai/liut61/val_study_commondisease10.json', 'r') as f:
             val_studies = json.load(f)
        # with open('/projects/site/gred/resbioai/liut61/test_study_commondisease10.json', 'r') as f:
        #      val_studies = json.load(f)
    return val_studies

def get_default_test_studies(classify_mode='binary'):
 #    test_studies = ['e2a4a67f-6a18-431a-ab9c-6e77dd31cc80',
 # '2a79d190-a41e-4408-88c8-ac5c4d03c0fc',
 # '7d7cabfd-1d1f-40af-96b7-26a0825a306d',
 # '60358420-6055-411d-ba4f-e8ac80682a2e',
 # 'be21c2d1-2392-47d0-96fb-c625d115e0dc']
    if classify_mode == 'binary':
        test_studies = ['DS000010475', 'GSE122703', 'GSE149313', 'GSE167363', 'GSE163668']

    elif classify_mode == 'multilabel':
        with open('/projects/site/gred/resbioai/liut61/test_study_commondisease10.json', 'r') as f:
             test_studies = json.load(f)
        # with open('/projects/site/gred/resbioai/liut61/val_study_commondisease10.json', 'r') as f:
        #      test_studies = json.load(f)
    return test_studies


def get_disease_set():
    disease_set = ['healthy', 'COVID-19']
    
    return disease_set


def get_disease_set_multi():

    disease_set = ["Alzheimer's disease",
 'B-cell acute lymphoblastic leukemia',
 'COVID-19',
 "Crohn's disease",
 'healthy',
 'lung adenocarcinoma',
 'melanoma',
 'multiple myeloma',
 'multiple sclerosis']

    return disease_set

class scDatasetWrapper(Dataset):
    """
    Wrapper around scimilarity_gred.tiledb_sample_data_models.scDataset dataset.
    In particular:
        - It adds padding if needed
        - It converts the output to CellSample
    """

    def __init__(self, sc_dataset: scDataset_disease, pad_size: int, classify_mode = 'binary'):
        self.sc_dataset = sc_dataset
        self.pad_size = pad_size
        # with open("/projects/site/gred/resbioai/liut61/disease_loader.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)


        if classify_mode == 'binary':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_2label.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        elif classify_mode == 'multilabel':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap9.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        
        # with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)
        # with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap10.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)

        # self.label_key = {}
        # for key,value in self.disease_label_dict.items():
        #     self.label_key[value] = key
            
        with open("/projects/site/gred/resbioai/liut61/disease_gpt4_description_embeddings.pickle", 'rb') as handle:
            self.disease_emb_dict = pickle.load(handle)

        with open("/projects/site/gred/resbioai/liut61/tissue_loader_all.pickle", 'rb') as handle:
            self.tissue_label_dict = pickle.load(handle)

    def __getitem__(self, idx: int) -> CellSample:
        sample_i = self.sc_dataset[idx]
        sample_i_X = torch.tensor(sample_i[0])
        disease_label = sample_i[5][0]
        tissue_label = sample_i[4][0]
        # print(disease_label)
        disease_label = self.disease_label_dict[disease_label]
        tissue_label = self.tissue_label_dict[tissue_label]

        disease_emb = self.disease_emb_dict[sample_i[5][0]]
        # disease_label = 0
        pad = torch.zeros(self.pad_size, dtype=torch.bool)
        pad[:sample_i_X.shape[0]] = True

        sample_i_padded = torch.zeros(self.pad_size, sample_i_X.shape[1])
        sample_i_padded[:sample_i_X.shape[0], :] = sample_i_X

        # cell_sample = CellSample(x=sample_i_padded, pad=pad, disease_label=torch.tensor(disease_label) - 1)
        
        cell_sample = CellSample(x=sample_i_padded, pad=pad, disease_label=torch.tensor(disease_label), disease_emb=torch.FloatTensor(disease_emb), tissue_label=torch.tensor(tissue_label))
        return cell_sample

    def __len__(self) -> int:
        return len(self.sc_dataset)

class SampleCellsDataModuleCustom(LightningDataModule):
    """
    Convenience class around SampleCellsDataModule
    """

    def __init__(
            self,
            cell_tdb_uri: str = cell_tdb_uri,
            counts_tdb_uri: str = counts_tdb_uri,
            gene_tdb_uri: str = gene_tdb_uri,
            gene_order: str = gene_order,
            val_studies: Optional[List[str]] = None,
            test_studies: Optional[List[str]] = None,
            label_column: str = "celltype_id",
            study_column: str = "study",
            sample_column: str = "sample",
            tissue_column : str = 'tissue',
            disease_column : str = 'disease',
            batch_size: int = 1,
            sample_size: int = 100,
            num_workers: int = 0,
            pad_size: Optional[int] = None,
            precompute_val_test_masking: bool = False,
            disease_set: Optional[int] = None,
            handle_imbalance = False,
            classify_mode = 'binary',
            resample = False
    ):
        super().__init__()
        if val_studies is None:
            val_studies = get_default_val_studies(classify_mode = classify_mode)
        if test_studies is None:
            test_studies = get_default_test_studies(classify_mode = classify_mode)
        if disease_set is None:
            if classify_mode == 'binary':
                disease_set = get_disease_set()
            elif classify_mode == 'multilabel':
                disease_set = get_disease_set_multi()
        self.data_module = SampleCellsDataModule_disease(cell_tdb_uri=cell_tdb_uri, counts_tdb_uri=counts_tdb_uri,
                                                 gene_tdb_uri=gene_tdb_uri,
                                                 gene_order=gene_order, val_studies=val_studies, test_studies=test_studies,
                                                 label_id_column=label_column,
                                                 study_column=study_column,
                                                 sample_column=sample_column, batch_size=batch_size,
                                                 sample_size=sample_size,
                                                 num_workers=num_workers,
                                                 tissue_column = tissue_column,
                                                 disease_column = disease_column,
                                                disease_set = disease_set,
                                                handle_imbalance = handle_imbalance, classify_mode = classify_mode, resample = resample
                                                )
        self.pad_size = pad_size
        if self.pad_size is None:
            self.pad_size = sample_size
        self.precompute_val_test_masking = precompute_val_test_masking

        if self.precompute_val_test_masking:
            # TODO
            raise NotImplementedError

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = scDatasetWrapper(self.data_module.train_dataset, pad_size=self.pad_size, classify_mode=classify_mode)
        if self.data_module.val_dataset is not None:
            self.val_dataset = scDatasetWrapper(self.data_module.val_dataset, pad_size=self.pad_size, classify_mode=classify_mode)
        if self.data_module.test_dataset is not None:
            self.test_dataset = scDatasetWrapper(self.data_module.test_dataset, pad_size=self.pad_size, classify_mode=classify_mode)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=CellSample.collate
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=CellSample.collate
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=CellSample.collate
        )

