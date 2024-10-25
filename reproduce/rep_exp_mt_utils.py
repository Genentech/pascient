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

class CellClassifyModel(L.LightningModule):
    def __init__(self, num_genes, masking_strategy: Masking, lr: float = 1e-04, weight_decay: float = 0., dropout: float = 0., text_emb = False, attn = 'linear_attn', n_dim = 1024, classify_mode = 'binary', num_layers=2, label_smoothing=0.0, logit_adjustment=False, residual=False, batchnorm = False, include_tissue=False, uncertainty=False, noise_robust=False, label_weight = False):
        super().__init__()

        # automatically access hparams with self.hparams.XXX
        self.save_hyperparameters(
            ignore=['gene_to_cell_encoder', 'cell_to_cell_encoder', 'cell_to_output_encoder', 'masking_strategy'])

        self.num_genes = num_genes
        self.masking_strategy = masking_strategy
        # with open("/projects/site/gred/resbioai/liut61/disease_loader.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)
        # with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)

        # with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap10.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)
        if classify_mode == 'binary':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_2label.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
            self.disease_number = len(self.disease_label_dict)
        elif classify_mode == 'multilabel':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap9.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle) 
            # self.disease_number = len(self.disease_label_dict) - 1
                self.disease_number = len(self.disease_label_dict)

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
            self.gene_to_cell_encoder = GeneToCellMLP(self.num_genes, latent_dim=n_dim, hidden_dim = [512,512], residual=residual, batchnorm=False, dropout = dropout)
        else:
            self.gene_to_cell_encoder = GeneToCellLinear(self.num_genes, latent_dim=n_dim)

        if self.include_tissue != False:
            self.cell_to_output_tissue_encoder = CellToOutputMLP(input_dim=n_dim , output_dim=self.tissue_number, hidden_dim=[512,512], dropout=dropout,residual=residual, batchnorm=batchnorm) # can be 512, 512
        
        if attn  == 'transformer':
            self.cell_to_cell_encoder = CellToCellPytorchTransformer(n_dim , n_heads=4, num_layers=num_layers, single_cell_only=False)
            
        self.cell_to_output_encoder = CellToOutputMLP(input_dim=n_dim , output_dim=self.disease_number, hidden_dim=[512,512], dropout=dropout,residual=residual, batchnorm=batchnorm) # can be 512, 512

        if logit_adjustment:
            if self.disease_number == 10:
                self.logit_adj = torch.load("/projects/site/gred/resbioai/liut61/adjusted_logits_data.pkl")
                logit_adj_list = list(self.logit_adj)
                logit_adj_list.insert(4, -10)
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
            
        if attn == 'nonlinear_attn':
            self.attention = nn.Sequential(
                nn.Linear(n_dim , n_dim ), # matrix V
                nn.Tanh(),
                nn.Linear(n_dim, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
            )


        self.loss_func = self.get_loss_func(label_smoothing = label_smoothing)
        self.metrics = self.get_metrics()

        self.noise_robust = noise_robust

    def get_loss_func(self, label_smoothing) -> nn.Module:
  #       return nn.CrossEntropyLoss(weight = torch.tensor([ 0.024203821656050957, 0.0021231422505307855,   0.10912951167728238,
  #  0.02505307855626327,    0.1851380042462845,     0.608067940552017,
  #  0.02038216560509554,  0.008492569002123142,  0.004670912951167728,
  # 0.012738853503184714]))
        if self.label_weight == False:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss(weight = 1.0 / torch.tensor([57,5, 268, 59, 1443, 48, 20, 11, 30]) , label_smoothing=label_smoothing)
    

    def get_metrics(self) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()
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
        x = batch.x
        y_label = batch.disease_label
        if (prefix == 'train') and self.noise_robust:
            x = torch.cat((x, x + torch.randn(x.shape).to(self.device)), 0)
            y_label = torch.cat((y_label,y_label), 0)
        
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
            
        o = self.cell_to_output_encoder(o_)  # sample x logits

        if self.logit_adjustment:
            o = o + self.logit_adj.to(o.device)
        
        preds = o
        gt = y_label # change it to y as labels
        loss = self.compute_loss(preds, gt)

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

    def obtain_embeddings_zs(self, x):
        o = self.gene_to_cell_encoder(x) 
        o = self.weighted_average(o)
        return o

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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer