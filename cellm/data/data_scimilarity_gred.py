from collections import Counter
import networkx as nx
import numpy as np
import obonet
import os, re
import pandas as pd
import pytorch_lightning as pl
import random
from scipy.sparse import coo_matrix, vstack
import tiledb
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Optional
import time
import pickle
from imblearn.over_sampling import RandomOverSampler ,SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler

from scimilarity_gred.tiledb_sample_data_models import query_tiledb_df, import_cell_ontology, get_id_mapper, scDataset


# Turn off multithreading to allow multiple pytorch dataloader workers
config = tiledb.Config()
config["sm.compute_concurrency_level"] = 1
config["sm.io_concurrency_level"] = 1
config["sm.num_async_threads"] = 1
config["sm.num_reader_threads"] = 1
config["sm.num_tbb_threads"] = 1
config["sm.num_writer_threads"] = 1
config["vfs.num_threads"] = 1


class scDataset_disease(Dataset):
    """A class that represent a collection of sample cells in TileDB."""

    def __init__(
        self,
        data_df: "pandas.DataFrame",
        counts_tdb: "tiledb.libtiledb.SparseArrayImpl",
        counts_shape: tuple,
        gene_indices: List[int],
        sample2cells: dict,
        int2sample: dict,
        sample_size: int,
        lognorm: bool = True,
        target_sum: float = 1e4,
        label_column: str = "cellTypeName",
        study_column: str = "dataset",
        sample_column: str = "sampleID",
        sparse: bool = False,
        tissue_column: str = 'tissue',
        disease_column: str = 'disease',
        handle_imbalance: bool = False,
        resample: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        data_df: pandas.DataFrame
            Pandas dataframe of valid cells.
        counts_tdb: tiledb.libtiledb.SparseArrayImpl
            Counts TileDB.
        gene_indices: List[int]
            The index of genes to return.
        sample_size: int
            Number of sample cells.
        lognorm: bool, default: True
            Whether to return log normalized expression instead of raw counts.
        target_sum: float, default: 1e4
            Target sum for log normalization.
        label_column: str, default: "cellTypeName"
            Label column name.
        study_column: str, default: "dataset"
            Study column name.
        sparse: bool, default: False
            Return a sparse matrix.
        """

        self.data_df = data_df
        self.counts_tdb = counts_tdb
        self.gene_indices = gene_indices
        self.sample2cells = sample2cells
        self.int2sample = int2sample
        self.sample_size = sample_size
        self.lognorm = lognorm
        self.target_sum = target_sum
        self.counts_shape = counts_shape
        self.label_column = label_column
        self.study_column = study_column
        self.sample_column = sample_column
        self.sparse = sparse
        self.tissue_column = tissue_column
        self.disease_column = disease_column

        self.resample = resample
        print(len(self.int2sample))
        if (handle_imbalance == True) or (handle_imbalance == 'disease tissue') or (handle_imbalance == 'disease tissue clean'):
            disease_label_list = []
            key_list = []
            for key,value in self.int2sample.items():
                sample = self.int2sample[key]
                cell_idx = self.sample2cells[sample]
                disease_label_list.append(self.data_df.loc[cell_idx, self.disease_column].values[0])
                key_list.append(key)
            ros = RandomOverSampler(random_state=0) # this method is changable
            train_data,train_label = ros.fit_resample(np.array(key_list).reshape(-1,1), disease_label_list)
            int2sample_new = {}
            for idx, k in enumerate(train_data):
                int2sample_new[idx] = self.int2sample[k[0]]
            self.int2sample = int2sample_new.copy()
            print(len(self.int2sample))

        if (handle_imbalance == 'disease downsample'):
            disease_label_list = []
            key_list = []
            for key,_ in self.int2sample.items():
                sample = self.int2sample[key]
                cell_idx = self.sample2cells[sample]
                disease_label_list.append(self.data_df.loc[cell_idx, self.disease_column].values[0])
                key_list.append(key)
            ros = RandomUnderSampler(random_state=0) # this method is changable
            train_data,train_label = ros.fit_resample(np.array(key_list).reshape(-1,1), disease_label_list)
            int2sample_new = {}
            for idx, k in enumerate(train_data):
                int2sample_new[idx] = self.int2sample[k[0]]
            self.int2sample = int2sample_new.copy()
            print(len(self.int2sample))

        # this method cannot work :) because of oversampled data are interpolated into original datasets.
        if handle_imbalance == 'disease svmsmote':
            disease_label_list = []
            key_list = []
            data_list = np.zeros((len(self.int2sample),len(self.gene_indices)))
            disease_match_dict = {}
            for key,value in self.int2sample.items():
                sample = self.int2sample[key]
                cell_idx = self.sample2cells[sample]
                if self.resample:
                    cell_idx = random.choices(self.sample2cells[sample], k=self.sample_size)
                else:
                    if len(cell_idx) < self.sample_size:
                        cell_idx = random.sample(self.sample2cells[sample], len(cell_idx))
                    else:
                        cell_idx = random.sample(self.sample2cells[sample], self.sample_size)

                results = self.counts_tdb.multi_index[cell_idx, :]
                counts = coo_matrix(
                    (results["vals"], (results["x"], results["y"])),
                    shape=self.counts_shape,
                ).tocsr()
                counts = counts[cell_idx, :]
                counts = counts[:, self.gene_indices]

                X = counts.astype(np.float32)
                if self.lognorm:
                    counts_per_cell = counts.sum(axis=1) + 1e-8
                    counts_per_cell = np.ravel(counts_per_cell)
                    counts_per_cell = counts_per_cell / self.target_sum
                    X = X / counts_per_cell[:, None]
                    X = X.log1p()
                data_list[key,:] = np.array(X.mean(axis=0)[0])
                disease_match_dict[tuple(np.round(data_list[key,:], 2))] = key
                disease_label_list.append(self.data_df.loc[cell_idx, self.disease_column].values[0])
                key_list.append(key)
            print(list(disease_match_dict.keys())[0])
            ros = SVMSMOTE(random_state=0, k_neighbors=4) # this method is changable
            train_data,train_label = ros.fit_resample(data_list, disease_label_list)
            int2sample_new = {}
            for idx, k in enumerate(train_data):
                int2sample_new[idx] = self.int2sample[disease_match_dict[tuple(np.round(k,2))]]
            self.int2sample = int2sample_new.copy()
            print(len(self.int2sample))

        if (handle_imbalance == 'disease tissue') or (handle_imbalance == 'tissue') or (handle_imbalance == 'tissue clean'):
            ##consider tissue resampling
            tissue_label_list = []
            key_list = []
            for key,value in self.int2sample.items():
                sample = self.int2sample[key]
                cell_idx = self.sample2cells[sample]
                tissue_label_list.append(self.data_df.loc[cell_idx, self.tissue_column].values[0])
                key_list.append(key)
            ros = RandomOverSampler(random_state=0) # this method is changable
            train_data,train_label = ros.fit_resample(np.array(key_list).reshape(-1,1), tissue_label_list)
            int2sample_new = {}
            for idx, k in enumerate(train_data):
                int2sample_new[idx] = self.int2sample[k[0]]
            self.int2sample = int2sample_new.copy()
            print(len(self.int2sample))

        if handle_imbalance == 'disease tissue clean':
            with open("/projects/site/gred/resbioai/liut61/tissue_clean_map.pickle", 'rb') as handle:
                self.tissue_label_map = pickle.load(handle)
            ##consider tissue resampling
            tissue_label_list = []
            key_list = []
            for key,value in self.int2sample.items():
                sample = self.int2sample[key]
                cell_idx = self.sample2cells[sample]
                tissue_label_list.append(self.tissue_label_map[self.data_df.loc[cell_idx, self.tissue_column].values[0]])
                key_list.append(key)
            ros = RandomOverSampler(random_state=0) # this method is changable
            train_data,train_label = ros.fit_resample(np.array(key_list).reshape(-1,1), tissue_label_list)
            int2sample_new = {}
            for idx, k in enumerate(train_data):
                int2sample_new[idx] = self.int2sample[k[0]]
            self.int2sample = int2sample_new.copy()
            print(len(self.int2sample))


    def __len__(self):
        return len(self.int2sample)

    def __getitem__(self, idx):
        sample = self.int2sample[idx]
        cell_idx = self.sample2cells[sample]

        if self.resample:
            cell_idx = random.choices(self.sample2cells[sample], k=self.sample_size)
        else:
            if len(cell_idx) < self.sample_size:
                cell_idx = random.sample(self.sample2cells[sample], len(cell_idx))
            else:
                cell_idx = random.sample(self.sample2cells[sample], self.sample_size)

        results = self.counts_tdb.multi_index[cell_idx, :]
        if "vals" in results.keys():
            counts = coo_matrix(
                (results["vals"], (results["x"], results["y"])),
                shape=self.counts_shape,
            ).tocsr()
        else:
            counts = coo_matrix(
                (results["data"], (results["cell_index"], results["gene_index"])),
                shape=self.counts_shape,
            ).tocsr()
        counts = counts[cell_idx, :]
        counts = counts[:, self.gene_indices]

        X = counts.astype(np.float32)
        if self.lognorm:
            counts_per_cell = counts.sum(axis=1) + 1e-8
            counts_per_cell = np.ravel(counts_per_cell)
            counts_per_cell = counts_per_cell / self.target_sum
            X = X / counts_per_cell[:, None]
            X = X.log1p()

        if not self.sparse:
            X = X.toarray()

        return (
            X,
            self.data_df.loc[cell_idx, self.label_column].values,
            self.data_df.loc[cell_idx, self.study_column].values,
            self.data_df.loc[cell_idx, self.sample_column].values,
            self.data_df.loc[cell_idx, self.tissue_column].values,
            self.data_df.loc[cell_idx, self.disease_column].values,
        )

class SampleCellsDataModule_disease(pl.LightningDataModule):
    """A class to encapsulate sample cells in TileDB to train the model."""

    def __init__(
        self,
        cell_tdb_uri: str,
        counts_tdb_uri: str,
        gene_tdb_uri: str,
        gene_order: str,
        val_studies: Optional[List[str]] = None,
        test_studies: Optional[List[str]] = None,
        label_id_column: str = "celltype_id",
        study_column: str = "study",
        sample_column: str = "sample",
        batch_size: int = 1,
        sample_size: int = 100,
        num_workers: int = 0,
        sparse: bool = False,
        min_sample_size: Optional[int] = None,
        tissue_column: str = 'tissue',
        disease_column: str = 'disease',
        disease_set: Optional[List[str]] = None,
        simple_mode: bool = False,
        handle_imbalance: bool = False,
        classify_mode: str = 'binary',
        resample: bool = False,
        nan_string: str = "",
        gene_name: str = 'genes',
    ):
        """Constructor.

        Parameters
        ----------
        cell_tdb_uri: str
            Cell metadata TileDB storage URI.
        counts_tdb_uri: str
            Counts TileDB storage URI.
        gene_tdb_uri: str
            Gene metadata TileDB storage URI.
        gene_order: str
            Use a given gene order as described in the specified file. One gene symbol per line.
        val_studies: List[str], optional, default: None
            List of studies to use as validation and test.
        label_id_column: str, default: "cellTypeOntologyID"
            Cell ontology ID column name.
        study_column: str, default: "dataset"
            Study column name.
        sample_column: str, default: "sampleID"
            Sample column name.
        batch_size: int, default: 1
            Batch size.
        sample_size: int, default: 100
            Sample size
        num_workers: int, default: 1
            The number of worker threads for dataloaders
        sparse: bool, default: False
            Use sparse matrices.
        min_sample_size: int, optional, default: None
            Set a minimum number of cells in a sample for it to be valid.

        Examples
        --------
        >>> datamodule = MetricLearningZarrDataModule(
                cell_tdb_uri="human_scref_cell_metadata",
                counts_tdb_uri="human_scref_counts_matrix",
                gene_tdb_uri="human_scref_gene_metadata",
                gene_order="gene_order.tsv",
                label_id_column="cellTypeOntologyID",
                batch_size=1000,
                num_workers=1,
            )
        """

        super().__init__()
        self.cell_tdb_uri = cell_tdb_uri
        self.counts_tdb_uri = counts_tdb_uri
        self.gene_tdb_uri = gene_tdb_uri
        self.val_studies = val_studies
        self.label_id_column = label_id_column
        self.study_column = study_column
        self.sample_column = sample_column
        self.tissue_column = tissue_column 
        self.disease_column = disease_column
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.min_sample_size = min_sample_size
        self.num_workers = num_workers
        self.sparse = sparse
        self.test_studies = test_studies
        self.disease_set = disease_set
        self.handle_imbalance =  handle_imbalance

        self.cell_tdb = tiledb.open(self.cell_tdb_uri, "r")
        self.counts_tdb = tiledb.open(self.counts_tdb_uri, "r", config=config)
        self.gene_tdb = tiledb.open(self.gene_tdb_uri, "r")
        self.simple_mode = simple_mode
        self.classify_mode = classify_mode

        self.resample = resample

        self.counts_shape = (self.cell_tdb.df[:].shape[0], self.gene_tdb.df[:].shape[0])

        # limit to cells with counts and labels
        if self.simple_mode:
            if self.classify_mode  == 'binary_ct':
                query_condition = f"{self.label_id_column} !=  '{nan_string}' and total_counts > 0"
                self.data_df = query_tiledb_df(
                    self.cell_tdb,
                    query_condition,
                    attrs=[self.study_column, self.sample_column, self.label_id_column, self.tissue_column, self.disease_column],
                )
            else:
                query_condition = f"{self.label_id_column} !=  '{nan_string}'" # we do not filter tissues.
                self.data_df = query_tiledb_df(
                    self.cell_tdb,
                    query_condition,
                    attrs=[self.study_column, self.sample_column, self.label_id_column, self.tissue_column, self.disease_column],
                )
        else:
            if self.classify_mode  == 'binary':
                query_condition = f"{self.tissue_column} == 'blood'"
            elif self.classify_mode == 'multilabel':
                query_condition = f"total_counts > 0" # we do not filter tissues.
            elif self.classify_mode == 'multilabel_final':
                query_condition = f"total_counts > 0" # we do not filter tissues.
            self.data_df = query_tiledb_df(
                self.cell_tdb,
                query_condition,
                attrs=[self.study_column, self.sample_column, self.label_id_column, self.tissue_column, self.disease_column],
            )

        # filter out sample without enough cells
        if self.min_sample_size is not None:
            self.study_sample_df = self.data_df.groupby(
                [self.study_column, self.sample_column]
            ).size()
            self.study_sample_df = (
                self.study_sample_df[self.study_sample_df >= self.min_sample_size]
                .copy()
                .reset_index()
            )
            self.data_df = self.data_df[
                self.data_df[self.study_column].isin(
                    self.study_sample_df[self.study_column]
                )
                & self.data_df[self.sample_column].isin(
                    self.study_sample_df[self.sample_column]
                )
            ].copy()

        # re_null = re.compile(pattern='\x00')
        result = self.data_df
        if self.disease_set is not None:
            print(self.disease_set)
            result = result.loc[result[self.disease_column].isin(self.disease_set)]

        # result = result.replace(regex=re_null, value=np.nan)
        if self.simple_mode:
            result = result.dropna() #we do not use cells without cell types.
        self.data_df = result 
        
        # map celltype ID to celltype name
        self.label_name_column = "cellTypeName"
        onto = import_cell_ontology()
        id2name = get_id_mapper(onto)
        self.data_df[self.label_name_column] = self.data_df[self.label_id_column].map(
            id2name
        )
    
        if self.simple_mode:
            self.data_df = self.data_df.dropna()

        # concat study and sample in the case there are duplicate sample names
        self.sampleID = "study-sample"
        self.data_df[self.sampleID] = (
            self.data_df[self.study_column] + self.data_df[self.sample_column]
        )

        self.val_df = None
        if self.val_studies is not None:
            # split out validation studies
            self.val_df = self.data_df[
                self.data_df[self.study_column].isin(self.val_studies)
            ]
            self.data_df = self.data_df[
                ~self.data_df[self.study_column].isin(self.val_studies)
            ]
            # limit validation celltypes to those in the training data
            self.val_df = self.val_df[
                self.val_df[self.label_name_column].isin(
                    self.data_df[self.label_name_column].unique()
                )
            ]

        self.test_df = None
        if self.test_studies is not None:
            # split out validation studies
            self.test_df = self.data_df[
                self.data_df[self.study_column].isin(self.test_studies)
            ]
            self.data_df = self.data_df[
                ~self.data_df[self.study_column].isin(self.test_studies)
            ]
            # limit validation celltypes to those in the training data
            self.test_df = self.test_df[
                self.test_df[self.label_name_column].isin(
                    self.data_df[self.label_name_column].unique()
                )
            ]
        print(f"Training data size: {self.data_df.shape}")
        if self.val_df is not None:
            print(f"Validation data size: {self.val_df.shape}")
        if self.test_df is not None:
            print(f"Testing data size: {self.test_df.shape}")

        self.class_names = set(self.data_df[self.label_name_column].values)
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {value: key for key, value in self.label2int.items()}

        # gene space needs be aligned to the given gene order
        with open(gene_order, "r") as fh:
            self.gene_order = [line.strip() for line in fh]
        if gene_name == 'genes':
            genes = self.gene_tdb.df[:]["genes"].tolist()
        if gene_name == 'cellarr_gene_index':
            genes = (
                    self.gene_tdb.query(attrs=["cellarr_gene_index"])
                    .df[:]["cellarr_gene_index"]
                    .tolist()
                )
        
        self.gene_indices = []
        for x in self.gene_order:
            try:
                self.gene_indices.append(genes.index(x))
            except:
                print(f"Gene not found: {x}")
                pass
        self.n_genes = len(self.gene_indices)  # used when creating training model

        gp = self.data_df.groupby(self.sampleID)
        self.train_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
        self.train_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
        self.train_dataset = scDataset_disease(
            data_df=self.data_df,
            counts_tdb=self.counts_tdb,
            counts_shape=self.counts_shape,
            gene_indices=self.gene_indices,
            sample2cells=self.train_sample2cells,
            int2sample=self.train_int2sample,
            sample_size=self.sample_size,
            label_column=self.label_name_column,
            study_column=self.study_column,
            sample_column=self.sample_column,
            sparse=self.sparse,
            tissue_column = self.tissue_column,
            disease_column = self.disease_column,
            handle_imbalance = self.handle_imbalance,
            resample = self.resample
        )

        self.val_dataset = None
        if self.val_df is not None:
            gp = self.val_df.groupby(self.sampleID)
            self.val_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
            self.val_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
            self.val_dataset = scDataset_disease(
            data_df=self.val_df,
            counts_tdb=self.counts_tdb,
            counts_shape=self.counts_shape,
            gene_indices=self.gene_indices,
            sample2cells=self.val_sample2cells,
            int2sample=self.val_int2sample,
            sample_size=self.sample_size,
            label_column=self.label_name_column,
            study_column=self.study_column,
            sample_column=self.sample_column,
            sparse=self.sparse,
            tissue_column = self.tissue_column,
            disease_column = self.disease_column,
            handle_imbalance = False,
            resample = self.resample
        )

        self.test_dataset = None
        if self.test_df is not None:
            gp = self.test_df.groupby(self.sampleID)
            self.test_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
            self.test_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
            self.test_dataset = scDataset_disease(
            data_df=self.test_df,
            counts_tdb=self.counts_tdb,
            counts_shape=self.counts_shape,
            gene_indices=self.gene_indices,
            sample2cells=self.test_sample2cells,
            int2sample=self.test_int2sample,
            sample_size=self.sample_size,
            label_column=self.label_name_column,
            study_column=self.study_column,
            sample_column=self.sample_column,
            sparse=self.sparse,
            tissue_column = self.tissue_column,
            disease_column = self.disease_column,
            handle_imbalance = False,
            resample = self.resample
        )

    def collate(self, batch):
        """Collate tensors.

        Parameters
        ----------
        batch:
            Batch to collate.

        Returns
        -------
        tuple
            A Tuple[torch.Tensor, torch.Tensor, list] containing information
            on the collated tensors.
        """

        profiles, labels, studies, samples = tuple(
            map(list, zip(*batch))
        )  # tuple([list(t) for t in zip(*batch)])
        if self.sparse:
            profiles = vstack(profiles)
            _c = torch.sparse_coo_tensor(
                [profiles.row, profiles.col], profiles.data, profiles.shape
            )
            profiles = _c.to_sparse_csr()
        else:
            profiles = torch.squeeze(torch.Tensor(np.vstack(profiles)))
        return (
            profiles,
            [x for l in labels for x in l],
            [x for l in studies for x in l],
            [x for l in samples for x in l],
        )

    def train_dataloader(self) -> DataLoader:
        """Load the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the training dataset.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate,
        )


    def val_dataloader(self) -> DataLoader:
        """Load the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the validation dataset.
        """

        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """
        if self.test_dataset is None:
            return self.val_dataloader()

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate,
        )