import os
from typing import List, Optional

import torch
from lightning import LightningDataModule
from scimilarity_gred.tiledb_sample_data_models import SampleCellsDataModule_disease, scDataset_disease
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
    celltype_label: torch.Tensor

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


tiledb_base_path = '/gstore/data/omni/scdb/tiledb'

CELLURI = "scimilarity_human_10x_cell_metadata"
GENEURI = "scimilarity_human_10x_gene_metadata"
COUNTSURI = "scimilarity_human_10x_counts"

cell_tdb_uri = os.path.join(tiledb_base_path, CELLURI)
gene_tdb_uri = os.path.join(tiledb_base_path, GENEURI)
counts_tdb_uri = os.path.join(tiledb_base_path, COUNTSURI)

gene_order = "/gstore/data/omni/scdb/cleaned_h5ads/gene_order.tsv"


def get_default_val_studies(classify_mode='binary', simple_mode = False):

    if (simple_mode) and (classify_mode == 'binary'):
        val_studies = ['9b02383a-9358-4f0f-9795-a891ec523bcc',
     'fcb3d1c1-03d2-41ac-8229-458e072b7a1c',
     'a98b828a-622a-483a-80e0-15703678befd',
     '29f92179-ca10-4309-a32b-d383d80347c1',
     '8191c283-0816-424b-9b61-c3e1d6258a77',
     'b3e2c6e3-9b05-4da9-8f42-da38a664b45b',
     'DS000010060',
     'DS000011735',
     '24d42e5e-ce6d-45ff-a66b-a3b3b715deaf',
     'DS000010475']
        return val_studies

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
    elif classify_mode == 'multilabel_final':
        with open('/projects/site/gred/resbioai/liut61/val_study_all.json', 'r') as f:
             val_studies = json.load(f)
    elif classify_mode == 'binary_ct':
        with open('/projects/site/gred/resbioai/liut61/val_study_2diseasect.json', 'r') as f:
             val_studies = json.load(f)
        # with open('/projects/site/gred/resbioai/liut61/test_study_commondisease10.json', 'r') as f:
        #      val_studies = json.load(f)
    return val_studies

def get_default_test_studies(classify_mode='binary', simple_mode = False):

    if (simple_mode) and (classify_mode == 'binary'):
        test_studies = ['e2a4a67f-6a18-431a-ab9c-6e77dd31cc80',
     '2a79d190-a41e-4408-88c8-ac5c4d03c0fc',
     '7d7cabfd-1d1f-40af-96b7-26a0825a306d',
     '60358420-6055-411d-ba4f-e8ac80682a2e',
     'be21c2d1-2392-47d0-96fb-c625d115e0dc']
        return test_studies
    if classify_mode == 'binary':
        test_studies = ['DS000010475', 'GSE122703', 'GSE149313', 'GSE167363', 'GSE163668']

    elif classify_mode == 'multilabel':
        with open('/projects/site/gred/resbioai/liut61/test_study_commondisease10.json', 'r') as f:
             test_studies = json.load(f)
    elif classify_mode == 'multilabel_final':
        with open('/projects/site/gred/resbioai/liut61/test_study_all.json', 'r') as f:
             test_studies = json.load(f)
    elif classify_mode == 'binary_ct':
        with open('/projects/site/gred/resbioai/liut61/test_study_2diseasect.json', 'r') as f:
             test_studies = json.load(f)
        # with open('/projects/site/gred/resbioai/liut61/val_study_commondisease10.json', 'r') as f:
        #      test_studies = json.load(f)
    return test_studies


def get_disease_set():
    disease_set = ['healthy', 'COVID-19']
    
    return disease_set


def get_disease_set_multi():

#     disease_set = ["Alzheimer's disease",
#  'B-cell acute lymphoblastic leukemia',
#  'COVID-19',
#  "Crohn's disease",
#  # 'None',
#  #  '',
#  'healthy',
#  'lung adenocarcinoma',
#  'melanoma',
#  'multiple myeloma',
#  'multiple sclerosis'
# ]

    disease_set = [
 'COVID-19',
 "Crohn's disease",
"Alzheimer's disease",
"lung adenocarcinoma",
"multiple sclerosis",
"melanoma",
'multiple myeloma',
'B-cell acute lymphoblastic leukemia',
  # '',
 'healthy'
    ]
    
    return disease_set

def get_disease_set_final():

    with open('/projects/site/gred/resbioai/liut61/disease_all.json', 'r') as f:
         disease_set = json.load(f)
    
    return disease_set


class scDatasetWrapper(Dataset):
    """
    Wrapper around scimilarity_gred.tiledb_sample_data_models.scDataset dataset.
    In particular:
        - It adds padding if needed
        - It converts the output to CellSample
    """

    def __init__(self, sc_dataset: scDataset_disease, pad_size: int, classify_mode = 'binary', tissue_clean = False):
        self.sc_dataset = sc_dataset
        self.pad_size = pad_size
        # with open("/projects/site/gred/resbioai/liut61/disease_loader.pickle", 'rb') as handle:
        #     self.disease_label_dict = pickle.load(handle)


        if classify_mode == 'binary':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_2label.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        elif classify_mode == 'binary_ct':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_2label.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
            with open("/projects/site/gred/resbioai/liut61/celltype_infomap.pickle", 'rb') as handle:
                self.celltype_label_dict = pickle.load(handle)
        elif classify_mode == 'multilabel':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_overlap9.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        elif classify_mode == 'multilabel_final':
            with open("/projects/site/gred/resbioai/liut61/disease_loader_all.pickle", 'rb') as handle:
                self.disease_label_dict = pickle.load(handle)
        self.classify_mode = classify_mode
        
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

        if tissue_clean == 'clean':
            with open("/projects/site/gred/resbioai/liut61/tissue_clean_loader.pickle", 'rb') as handle:
                self.tissue_label_dict = pickle.load(handle)

            with open("/projects/site/gred/resbioai/liut61/tissue_clean_map.pickle", 'rb') as handle:
                self.tissue_label_map = pickle.load(handle)

        self.tissue_clean = tissue_clean

    def __getitem__(self, idx: int) -> CellSample:
        sample_i = self.sc_dataset[idx]
        sample_i_X = torch.tensor(sample_i[0])
        disease_label = sample_i[5][0]
        tissue_label = sample_i[4][0]
        # print(disease_label)
        # print(tissue_label)
        disease_label = self.disease_label_dict[disease_label]
        if self.tissue_clean == False:
            tissue_label = self.tissue_label_dict[tissue_label]
        else:
            tissue_label = self.tissue_label_dict[self.tissue_label_map[tissue_label]]
            
        disease_emb = self.disease_emb_dict[sample_i[5][0]]
        # disease_label = 0
        pad = torch.zeros(self.pad_size, dtype=torch.bool)
        pad[:sample_i_X.shape[0]] = True

        sample_i_padded = torch.zeros(self.pad_size, sample_i_X.shape[1])
        sample_i_padded[:sample_i_X.shape[0], :] = sample_i_X
        if self.classify_mode == 'binary_ct':
            celltype_label = [self.celltype_label_dict[i] for i in sample_i[1]]
        else:
            celltype_label = tissue_label
            
        # cell_sample = CellSample(x=sample_i_padded, pad=pad, disease_label=torch.tensor(disease_label) - 1)
        
        cell_sample = CellSample(x=sample_i_padded, pad=pad, disease_label=torch.tensor(disease_label), disease_emb=torch.FloatTensor(disease_emb), tissue_label=torch.tensor(tissue_label), celltype_label = torch.tensor(celltype_label))
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
            resample = False,
            tissue_clean = False,
            gene_name = 'genes',
            simple_mode = False
    ):
        super().__init__()
        if val_studies is None:
            val_studies = get_default_val_studies(classify_mode = classify_mode, simple_mode=simple_mode)
        if test_studies is None:
            test_studies = get_default_test_studies(classify_mode = classify_mode, simple_mode=simple_mode)
        if disease_set is None:
            if (classify_mode == 'binary') or (classify_mode == 'binary_ct'):
                disease_set = get_disease_set()
            elif classify_mode == 'multilabel':
                disease_set = get_disease_set_multi()
            elif classify_mode == 'multilabel_final':
                disease_set = get_disease_set_final()
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
                                                handle_imbalance = handle_imbalance, classify_mode = classify_mode, resample = resample, gene_name=gene_name, simple_mode=simple_mode
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

        self.train_dataset = scDatasetWrapper(self.data_module.train_dataset, pad_size=self.pad_size, classify_mode=classify_mode, tissue_clean=tissue_clean)
        if self.data_module.val_dataset is not None:
            self.val_dataset = scDatasetWrapper(self.data_module.val_dataset, pad_size=self.pad_size, classify_mode=classify_mode, tissue_clean=tissue_clean)
        if self.data_module.test_dataset is not None:
            self.test_dataset = scDatasetWrapper(self.data_module.test_dataset, pad_size=self.pad_size, classify_mode=classify_mode, tissue_clean=tissue_clean)


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
