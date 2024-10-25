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


# @dataclass
# class CellSample:
#     x: torch.Tensor
#     pad: torch.Tensor
#     disease_label: torch.Tensor
#     disease_emb: torch.Tensor

#     @staticmethod
#     def collate(batch):
#         device = batch[0].x.device
#         collated = {}
#         keys = batch[0].__dict__.keys()
#         for key in keys:
#             # print(key)
#             attribute_list = [getattr(b, key) for b in batch]
#             collated[key] = torch.stack(attribute_list).to(device=device)
#         return CellSample(**collated)

@dataclass
class CellSample:
    x: torch.Tensor
    pad: torch.Tensor
    disease_label: torch.Tensor
    disease_emb: torch.Tensor

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

def get_default_val_studies():
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
    
    # with open('/projects/site/gred/resbioai/liut61/val_study_commondisease10.json', 'r') as f:
    #      val_studies = json.load(f)
    return val_studies

def get_default_test_studies():
 #    test_studies = ['e2a4a67f-6a18-431a-ab9c-6e77dd31cc80',
 # '2a79d190-a41e-4408-88c8-ac5c4d03c0fc',
 # '7d7cabfd-1d1f-40af-96b7-26a0825a306d',
 # '60358420-6055-411d-ba4f-e8ac80682a2e',
 # 'be21c2d1-2392-47d0-96fb-c625d115e0dc']

    test_studies = ['DS000010475', 'GSE122703', 'GSE149313', 'GSE167363', 'GSE163668']
    # with open('/projects/site/gred/resbioai/liut61/test_study_commondisease10.json', 'r') as f:
    #      test_studies = json.load(f)
    return test_studies


def get_disease_set():
    disease_set = ['healthy', 'COVID-19']
 #    disease_set = ['ulcerative colitis',
 # 'viral encephalitis',
 # 'pancreatic carcinoma',
 # 'COVID-19;healthy',
 # 'juvenile idiopathic arthritis',
 # 'multiple sclerosis',
 # 'idiopathic pulmonary fibrosis',
 # 'atopic eczema',
 # 'non-alcoholic fatty liver disease',
 # 'healthy',
 # 'myocardial infarction',
 # 'monoclonal gammopathy',
 # 'multiple myeloma',
 # 'Tonsillar Squamous Cell Carcinoma',
 # 'basal cell carcinoma',
 # 'systemic lupus erythematosus',
 # 'cystic fibrosis',
 # "Crohn's disease",
 # 'type II diabetes mellitus',
 # 'chronic rhinosinusitis with nasal polyps',
 # 'adenocarcinoma',
 # 'Gastric Metaplasia',
 # 'neuroblastoma',
 # 'fibrosis',
 # 'B-cell lymphoma',
 # 'cutaneous squamous cell carcinoma',
 # 'head and neck squamous cell carcinoma',
 # 'hidradenitis suppurativa',
 # 'liver neoplasm;Uveal Melanoma',
 # 'Uveal Melanoma',
 # 'acute kidney failure',
 # 'drug hypersensitivity syndrome',
 # 'mucocutaneous lymph node syndrome',
 # 'chronic myelogenous leukemia',
 # 'lung adenocarcinoma',
 # 'chronic kidney disease',
 # 'colorectal cancer',
 # "Alzheimer's disease",
 # 'hepatocellular carcinoma',
 # 'medulloblastoma',
 # 'T-cell acute lymphoblastic leukemia',
 # 'Immune dysregulation-polyendocrinopathy-enteropathy-X-linked syndrome',
 # 'COVID-19',
 # 'interstitial lung disease',
 # 'None',
 # 'hypoplastic left heart syndrome',
 # 'chronic obstructive pulmonary disease',
 # 'dengue disease',
 # 'acute myeloid leukemia',
 # 'B-cell acute lymphoblastic leukemia',
 # 'Diamond-Blackfan anemia',
 # 'prostatic hypertrophy',
 # 'glioma',
 # 'dilated cardiomyopathy',
 # 'arrhythmogenic right ventricular cardiomyopathy',
 # 'melanoma',
 # 'gastric cancer',
 # 'intracranial hypotension',
 # "Parkinson's Disease",
 # 'pancreatic ductal adenocarcinoma']

 #    disease_set = ["Alzheimer's disease",
 # 'B-cell acute lymphoblastic leukemia',
 # 'COVID-19',
 # "Crohn's disease",
 # 'None',
 # 'healthy',
 # 'lung adenocarcinoma',
 # 'melanoma',
 # 'multiple myeloma',
 # 'multiple sclerosis']
    
    return disease_set


class scDatasetWrapper(Dataset):
    """
    Wrapper around scimilarity_gred.tiledb_sample_data_models.scDataset dataset.
    In particular:
        - It adds padding if needed
        - It converts the output to CellSample
    """

    def __init__(self, sc_dataset: scDataset_disease, pad_size: int):
        self.sc_dataset = sc_dataset
        self.pad_size = pad_size
        # with open("/projects/site/gred/resbioai/liut61/disease_loader.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)


        with open("/projects/site/gred/resbioai/liut61/disease_loader_2label.pickle", 'rb') as handle:
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

    def __getitem__(self, idx: int) -> CellSample:
        sample_i = self.sc_dataset[idx]
        sample_i_X = torch.tensor(sample_i[0])
        disease_label = sample_i[5][0]
        # print(disease_label)
        #global_count_sample.append(sample_i)
        disease_label = self.disease_label_dict[disease_label]

        disease_emb = self.disease_emb_dict[sample_i[5][0]]
        # disease_label = 0
        pad = torch.zeros(self.pad_size, dtype=torch.bool)
        pad[:sample_i_X.shape[0]] = True

        #zero padding
        # sample_i_padded = torch.zeros(self.pad_size, sample_i_X.shape[1])
        # sample_i_padded[:sample_i_X.shape[0], :] = sample_i_X

        
        sample_i_padded = sample_i_X.float()

        # cell_sample = CellSample(x=sample_i_padded, pad=pad, disease_label=torch.tensor(disease_label) - 1)
        
        cell_sample = CellSample(x=sample_i_padded, pad=pad, disease_label=torch.tensor(disease_label), disease_emb=torch.FloatTensor(disease_emb))
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
        simple_mode = False
    ):
        super().__init__()
        if val_studies is None:
            val_studies = get_default_val_studies()
        if test_studies is None:
            test_studies = get_default_test_studies()
        if disease_set is None:
            disease_set = get_disease_set()
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
                                                handle_imbalance = handle_imbalance, simple_mode = simple_mode
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

        self.train_dataset = scDatasetWrapper(self.data_module.train_dataset, pad_size=self.pad_size)
        if self.data_module.val_dataset is not None:
            self.val_dataset = scDatasetWrapper(self.data_module.val_dataset, pad_size=self.pad_size)
        if self.data_module.test_dataset is not None:
            self.test_dataset = scDatasetWrapper(self.data_module.test_dataset, pad_size=self.pad_size)

        self.save_ct_result = []


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
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


import numpy as np
f1score_list = []
finalauroc_list = []
from sklearn.metrics import classification_report #1e-3, best save
from sklearn.metrics import roc_auc_score

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule

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
# from cellm.data.data_structures import CellSample
import pickle

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
        

    def obtain_annotation(self, batch:CellSample, batch_index):
        o = self.general_compute(batch, batch_index)
        o = self.cell_to_output_encoder(o)  # batch x sample x cell
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
