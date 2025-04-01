import os
from typing import List, Optional, Tuple, Callable, Union, Dict

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Sampler

from pascient.data.batch_mappers import BatchMapper

from pascient.data.data_structures import CellSample

from pascient.data.data_augmentations import DataAugmentation

from pascient.data.data_samplers import BaseSampler

from pascient.components.misc import path_exists, save_sparse_matrix, load_sparse_matrix, remove_file

from scimilarity.utils import query_tiledb_df

from pascient.components.misc import PartialClass

from scimilarity.ontologies import import_cell_ontology, get_id_mapper
import pytorch_lightning as pl
import tiledb

import pandas as pd
import random
import numpy as np
from scipy.sparse import coo_matrix

import json
import sys

import logging
log = logging.getLogger(__name__)


def get_default_val_studies(dataset_name, val_studies_paths):
    """
    Returns the default validation partition for the dataset.
    The validation partition is a tuple with the name of the variable to partition by and a list of identifiers for the validation set.
    """
    if dataset_name == "scimilarity":
        val_studies = ('study',[
            "24d42e5e-ce6d-45ff-a66b-a3b3b715deaf",
            "29f92179-ca10-4309-a32b-d383d80347c1",
            "2a79d190-a41e-4408-88c8-ac5c4d03c0fc",
            "60358420-6055-411d-ba4f-e8ac80682a2e",
            "7d7cabfd-1d1f-40af-96b7-26a0825a306d",
            "8191c283-0816-424b-9b61-c3e1d6258a77",
            "9b02383a-9358-4f0f-9795-a891ec523bcc",
            "a98b828a-622a-483a-80e0-15703678befd",
            "b3e2c6e3-9b05-4da9-8f42-da38a664b45b",
            "be21c2d1-2392-47d0-96fb-c625d115e0dc",
            "DS000010060",
            "DS000010475",
            "DS000011735",
            "e2a4a67f-6a18-431a-ab9c-6e77dd31cc80",
            "fcb3d1c1-03d2-41ac-8229-458e072b7a1c",
        ])
    elif dataset_name == "AD":
        val_studies = ('sample_id', ['R5256488', 'R8678748', 'R5079327', 'H21.33.007', 'H20.33.043',
       'R1287407', 'H21.33.045', 'H200.1023', 'R3485645', 'R2393217',
       'R4990009', 'H20.33.032', 'R3368249', 'R4087581', 'R6698302',
       'R3740754', 'R5541746', 'R9537646', 'R7698313', 'R9393519',
       'R2645096', 'R4042599', 'R8760165', 'R4470253', 'R8057601',
       'R4415805', 'R9419876', 'R2045909', 'R2667630', 'R9794121',
       'R2787688', 'R1028639', 'R6337324', 'R9354381', 'R6934314',
       'R7915228', 'R3722356', 'R1133959', 'R7978618', 'R6415047',
       'R1042566', 'R1234575', 'R9309271', 'R2166876', 'R2895885',
       'R7295303', 'R3948425', 'R8330118', 'H21.33.020', 'R2424757',
       'R3607578', 'R5158294', 'R5676537', 'R4917253', 'R2830542',
       'R5079107', 'R9127940', 'H21.33.008', 'R6604064', 'R9680160',
       'R5574987', 'R9557117', 'R2006886', 'R8724814', 'R1218460',
       'H20.33.027', 'H20.33.036', 'H21.33.030', 'R6887989', 'R1620679',
       'H21.33.015', 'R4361022', 'R7095349', 'R6346298', 'R8472815',
       'R8553520', 'R3077672', 'H21.33.021', 'R7702934'])
    elif dataset_name == "pascient_binary":
        val_studies = ('study',['ddfad306-714d-4cc0-9985-d9072820c530',
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
                                'GSE128879'])
    elif dataset_name == "pascient_multilabel":
        with open(val_studies_paths['val'], 'r') as f:
            studies = json.load(f)
            studies += ['ddfad306-714d-4cc0-9985-d9072820c530']
            val_studies = ("study",studies)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return val_studies

def get_default_test_studies(dataset_name, val_studies_paths):
    """
    Returns the default test partition for the dataset.
    The test partition is a tuple with the name of the variable to partition by and a list of identifiers for the test set.
    """
    if dataset_name == "scimilarity":
        test_studies = ('study', ['DS000010475', 'GSE122703', 'GSE149313', 'GSE167363', 'GSE163668'])
    elif dataset_name == "AD":
        test_studies = ('sample_id',['R6289070', 'R5405023', 'R8608442', 'H21.33.027', 'R5177066',
       'R3857147', 'R3739042', 'R8293796', 'R3844037', 'R7063792',
       'R4012015', 'R1262106', 'R6728038', 'R3078606', 'R7054373',
       'R8998310', 'R6253512', 'R4900012', 'R4927046', 'R7791442',
       'R6802400', 'R2543886', 'H20.33.004', 'R3741788', 'R3111222',
       'H20.33.008', 'R3121235', 'R2678902', 'R5955028', 'R2575548',
       'R8125311', 'R3328752', 'R9245150', 'R6280004', 'R7912121',
       'R5234179', 'R1214999', 'R5026720', 'H21.33.041', 'R4935546',
       'R1672797', 'R3035452', 'R6759986', 'R7286984', 'H21.33.026',
       'R7993799', 'H21.33.029', 'H20.33.011', 'R5131375', 'R9596785',
       'R6622577', 'R5935442', 'H20.33.016', 'R1531359', 'H21.33.002',
       'H20.33.028', 'H21.33.039', 'R5789564', 'R7384738', 'R3744330',
       'R2793780', 'H21.33.046', 'H20.33.029', 'R5334541', 'R6392007',
       'R2879330', 'H21.33.010', 'R3211474', 'R2157677', 'R5459434',
       'R6665276', 'R2494273', 'H20.33.024', 'R3884524', 'R7551006',
       'R4641987', 'R5546461', 'R4323608', 'R4745715'])
    elif dataset_name == "pascient_binary":
        test_studies = ("study",['DS000010475', 'GSE122703', 'GSE149313', 'GSE167363', 'GSE163668'])
    elif dataset_name == "pascient_multilabel":
        with open(val_studies_paths['test'], 'r') as f:
            test_studies = ("study",json.load(f))
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return test_studies



class scDataset(Dataset):
    """A class that represent a collection of sample cells in TileDB."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        int2sample: dict,
        sample2cells: dict,
        sample_size: int,
        sampling_by_class: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        data_df: pandas.DataFrame
            Pandas dataframe of valid cells.
        int2sample: dict
            A mapping of sample index to sample id.
        sample2cells: dict
            A mapping of sample id to cell indices.
        sample_size: int
            Number of sample cells.
        sampling_by_class: bool, default: False
            Sample based on class counts, where sampling weight is inversely proportional to count.
        """

        self.int2sample = int2sample
        self.sample2cells = sample2cells
        self.sample_size = sample_size
        self.sampling_by_class = sampling_by_class
        self.data_df = data_df

    def __len__(self):
        return len(self.int2sample)

    def __getitem__(self, idx):
        sample = self.int2sample[idx]
        cell_idx = self.sample2cells[sample]

        if len(cell_idx) < self.sample_size:
            cell_idx = random.sample(self.sample2cells[sample], len(cell_idx))
        else:
            if self.sampling_by_class:
                sample_df = self.data_df.loc[self.sample2cells[sample], :].copy()
                sample_df = sample_df.sample(
                    n=self.sample_size, weights="sampling_weight"
                )
                cell_idx = sample_df.index.tolist()
            else:
                cell_idx = random.sample(self.sample2cells[sample], self.sample_size)

        return self.data_df.loc[cell_idx].copy()

    

class scDatasetAugmented(scDataset):
    """ Augmented version of scDataset that allows for multiple views per sample.

    Args:
        data_df: DataFrame with the cells metadata
        int2sample: dictionary mapping integer to sample id
        sample2cells: dictionary mapping sample id to cell indices
        sample_size: number of cells per sample
        n_views_per_sample: number of views per sample (augmentation)
        overlap_samples: whether to allow for overlapping samples (wether cells from same sample should be all different)
            if True, the views of each sample can potentially overlap.
    
    """
    def __init__(self,
                 data_df: pd.DataFrame,
                 int2sample: dict,
                 sample2cells: dict,
                 sample_size: int,
                 n_views_per_sample: int = 1,
                 overlap_samples: int = True):
        super().__init__(data_df = data_df, 
                         int2sample = int2sample,
                          sample2cells = sample2cells,
                          sample_size = sample_size)
        
        self.n_views_per_sample = n_views_per_sample
        self.overlap_samples = overlap_samples
        
    def __getitem__(self,idx):
        sample = self.int2sample[idx]
        cell_idx = self.sample2cells[sample]

        if self.overlap_samples: # cells can overlap over the different views
            df_list = []
            for i in range(self.n_views_per_sample):
                df_list.append(super().__getitem__(idx))
            
        else:
            if len(cell_idx) < self.n_views_per_sample * self.sample_size:
                cell_idx = random.sample(self.sample2cells[sample], len(cell_idx))
                cell_idx = np.array_split(cell_idx, self.n_views_per_sample)
                df_list = [self.data_df.loc[cell_idx_.tolist()].copy() for cell_idx_ in cell_idx]
            
            else:
                cell_idx = random.sample(self.sample2cells[sample], self.n_views_per_sample * self.sample_size)

                cell_idx = np.array_split(cell_idx, self.n_views_per_sample)
                df_list = [self.data_df.loc[cell_idx_.tolist()].copy() for cell_idx_ in cell_idx]
        
        df_lengths = [len(df) for df in df_list]
        view_id = np.repeat(np.arange(self.n_views_per_sample), df_lengths)
        df = pd.concat(df_list)
        df["view_id"] = view_id #adding a colum for the view id
        #breakpoint()

        #if df.view_id.nunique()<self.n_views_per_sample:
        #    breakpoint()
        
        return df

def worker_init_fn(worker_id):
    print(f"Worker {worker_id} is initialized")
        
class PatientCellsDataModule(pl.LightningDataModule):
    """
    Data module for sampling cells from individual patients / samples.

    Returns (normalized) gene counts and metadata df for each sample.
    """
    def __init__(
        self,
        cell_tdb_uri: str,
        gene_tdb_uri: str,
        matrix_tdb_uri: str,
        gene_order: str,
        output_map: BatchMapper,
        val_studies: Optional[List] = None,
        test_studies: Optional[List] = None,
        label_id_column: str = "celltype_id",
        study_column: str = "study",
        sample_column: str = "cellarr_sample",
        extra_columns: list = ["tissue", "disease"],
        gene_column: str = "genes",
        batch_size: int = 1,
        sample_size: int = 100,
        num_workers: int = 0,
        lognorm: bool = True,
        target_sum: float = 1e4,
        sparse: bool = False,
        min_sample_size: Optional[int] = None,
        nan_strings: List[str] = ['nan','<NA>'],
        sampler_cls= None, # Some issues with CLI to get the typing right
        remove_new_val_labels: bool = False,
        cached_db: str = None,
        overwrite_cache: bool = False,
        dataset_cls: Dataset = scDataset,
        augmentations: List[DataAugmentation] = None,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
        dataset_name:str = "scimilarity",
        val_studies_paths: Dict = None,
        num_genes:str = 0,
        oversampling:List[str] = None,
    ):
        """
        Parameters
        ----------
        cell_tdb_uri: str
            Cell metadata TileDB storage URI.
        gene_tdb_uri: str
            Gene metadata TileDB storage URI.
        matrix_tdb_uri: str
            Counts TileDB storage URI.
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
        gene_column: str, default: "genes"
            Gene column name in the gene tiledb.
        batch_size: int, default: 1
            Batch size.
        sample_size: int, default: 100
            Sample size
        num_workers: int, default: 1
            The number of worker threads for dataloaders
        lognorm: bool, default: True
            Whether to return log normalized expression instead of raw counts.
        target_sum: float, default: 1e4
            Target sum for log normalization.
        sparse: bool, default: False
            Use sparse matrices.
        sampling_by_class: bool, default: False
            Sample based on class counts, where sampling weight is inversely proportional to count.
            If False, use random sampling.
        remove_singleton_classes: bool, default: False
            Exclude cells with classes that exist in only one sample.
        min_sample_size: int, optional, default: None
            Set a minimum number of cells in a sample for it to be valid.
        nan_string: str, default: "nan"
            A string representing NaN.
        sampler_cls: Sampler, default: BaseBatchSampler
            Sampler class to use for batching.
        dataset_cls: Dataset, default: scDataset
            Base Dataset class to use.
        persistent_workers: bool, default: True
            If True, uses persistent workers in the DataLoaders.
        multiprocessing_context: str, default: "spawn"
            Multiprocessing context to use for the DataLoaders.
        output_map: Callable
            Function to map the output of the collate function. Default is None (Identity).
        cached_db: str
            Path to the cached db, for development purposes. Default is None (No cache).
        overwrite_cache: bool
            Whether to overwrite the cache. Default is False (Do not overwrite). Useful if one wants to generate new cache files.
        remove_new_val_labels: bool, default: False
            If True, the classes not contained in the training data will be excluded from the validation and test data.
        num_genes: int, default: 0
            Currently this is not used, but kept as an argument for compatibility. 
        oversampling: List, default: None
            List of attributes to oversample the data by (e.g ["tissue", "disease"])
        val_studies_path: Dict, default: None
            Path to the validation studies file. Default is None.
            Should have structure: {"val": "path/to/val_studies.json", "test": "path/to/test_studies.json"}
        """

        super().__init__()

        self.cell_tdb_uri = cell_tdb_uri
        self.gene_tdb_uri = gene_tdb_uri
        self.matrix_tdb_uri = matrix_tdb_uri
        self.val_studies = val_studies
        self.test_studies = test_studies
        self.label_id_column = label_id_column
        self.study_column = study_column
        self.sample_column = sample_column
        self.gene_column = gene_column
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_workers = num_workers
        self.lognorm = lognorm
        self.target_sum = target_sum
        self.sparse = sparse
        self.remove_new_val_labels = remove_new_val_labels
        self.min_sample_size = min_sample_size
        self.nan_strings = nan_strings
        self.sampler_cls = sampler_cls
        self.dataset_cls = dataset_cls
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.extra_columns = extra_columns
        self.output_map = output_map
        self.cached_db = cached_db
        self.cached_counts = None
        self.gene_order = gene_order
        self.dataset_name = dataset_name
        self.oversampling = oversampling

        if self.val_studies is None:
            self.val_studies = get_default_val_studies(dataset_name, val_studies_paths)
        if self.test_studies is None:
            self.test_studies = get_default_test_studies(dataset_name, val_studies_paths)

        if overwrite_cache: # remove the cached files to generate new ones.
            
            assert self.cached_db is not None, "Cached db path must be provided to overwrite cache...."

            try:
                remove_file(os.path.join(self.cached_db,"data_df.csv"))
                remove_file(os.path.join(self.cached_db,"counts_csr.npz"))
            except OSError as error:
                log.error(error)
                log.error("Cached files can not be removed - probably they do not exist.")

        self.cell_tdb = tiledb.open(self.cell_tdb_uri, "r")
        self.gene_tdb = tiledb.open(self.gene_tdb_uri, "r")
        #self.matrix_tdb = tiledb.open(self.matrix_tdb_uri, "r")

        self.matrix_shape = (
            self.cell_tdb.nonempty_domain()[0][1] + 1,
            self.gene_tdb.nonempty_domain()[0][1] + 1,
        )

        self.set_tile_db_config()
        
        self.filter_db()

        self.divide_data()

        #self.process_label_column()

        self.align_gene_space()

        self.create_datasets() 

        self.output_map.init_mapping(self.data_df, onto = self.onto)
        self.output_map.set_normalization(self.lognorm, self.target_sum)

        if self.cached_db is not None: #Retrieve from cache if it exists.
           self.load_counts_from_cache()

        # Process Augmentations:
        self.augmentations = {k: v(data_df = self.data_df) for k,v in augmentations.items()}
        self.output_map.set_augmentations(self.augmentations)

        self.cell_tdb.close()
        self.gene_tdb.close()

        self.load_samplers()

        self.cleanup()


    def set_tile_db_config(self):
        self.tdb_cfg = tiledb.Config()
        #cfg["sm.mem.total_budget"] = 50000000000  # 50G
        # turn off tiledb multithreading
        self.tdb_cfg["sm.compute_concurrency_level"] = 1
        self.tdb_cfg["sm.io_concurrency_level"] = 1
        self.tdb_cfg["sm.num_async_threads"] = 1
        self.tdb_cfg["sm.num_reader_threads"] = 1
        self.tdb_cfg["sm.num_tbb_threads"] = 1
        self.tdb_cfg["sm.num_writer_threads"] = 1
        self.tdb_cfg["vfs.num_threads"] = 1
        

    def cleanup(self):
        """
        Deleting large objects that are not needed after initialization
        """
        del self.data_df
        del self.train_df
        del self.val_df
        del self.test_df


    def load_samplers(self):
        
        self.train_sampler = self.sampler_cls(
            data_df=self.train_df,
            int2sample=self.train_int2sample,
            bsz=self.batch_size,
            drop_last=False,
            shuffle=True,
            data_split="train",
        )

        if self.val_dataset is not None:
            self.val_sampler = self.sampler_cls(
                data_df=self.val_df,
                int2sample=self.val_int2sample,
                bsz=self.batch_size,
                drop_last=False,
                shuffle=False,
                data_split = "val",
            )

        if self.test_dataset is not None:
            self.test_sampler = self.sampler_cls(
                data_df=self.test_df,
                int2sample=self.test_int2sample,
                bsz=self.batch_size,
                drop_last=False,
                shuffle=False,
                data_split="test",
            )

    def load_counts_from_cache(self):
        """Load counts from cache."""

        if not path_exists(os.path.join(self.cached_db,"counts_csr.npz")):
            log.info("Generating cache dataset...")
            results = self.matrix_tdb.multi_index[self.data_df.index.tolist(), :]
            counts = coo_matrix(
                (results["data"], (results["cell_index"], results["gene_index"])),
                shape=self.matrix_shape,
                ).tocsr()
            save_sparse_matrix(os.path.join(self.cached_db,"counts_csr.npz"), counts)
            log.info("Done")
        else:
            log.info("Fetching cached counts...")
            counts = load_sparse_matrix(os.path.join(self.cached_db,"counts_csr.npz"))
            log.info("Done")
        
        self.cached_counts = counts
        return
    
    def filter_db(self):
        """
        Filtering the tiledb.
        - Querying the tiledb and prefiltering (in query_db())
        - Filtering for studies with minimum number of cells
        - Mapping the cell-type to reference ontology map
        """
        
        self.query_db()

        # Filtering samples with less than 100 cells
        if self.min_sample_size is not None:
            log.info(f"Filtering samples with less than {self.min_sample_size} cells...")
            sample_sizes = self.data_df.groupby([self.sample_column]).size()
            sample_sizes = sample_sizes[sample_sizes >= 100]
            self.data_df = self.data_df.loc[self.data_df[self.sample_column].isin(sample_sizes.index)]


        # map celltype ID to celltype name
        if self.label_id_column == "celltype_id":
            self.label_name_column = "cellTypeName"
            self.onto = import_cell_ontology()
            self.id2name = get_id_mapper(self.onto)
            self.data_df[self.label_name_column] = self.data_df[self.label_id_column].map(
                self.id2name
            )
            self.data_df = self.data_df.dropna()
        else:
            self.label_name_column = self.label_id_column
            self.onto = None

        # concat study and sample in the case there are duplicate sample names
        self.sampleID = "study::::sample"
        self.data_df[self.sampleID] = (
            self.data_df[self.study_column] + "::::" + self.data_df[self.sample_column]
        ) 

    #def __del__(self):
    #    self.matrix_tdb.close()

    def create_datasets(self):

        log.info("Create Datasets...") 
        ### Remark: we may not need the label_int anymore...TBD.
        gp = self.train_df.groupby(self.sampleID)
        self.train_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
        self.train_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
        #self.train_df["label_int"] = self.train_df[self.label_name_column].map(
        #    self.label2int
        #)
        
        self.train_dataset = self.dataset_cls(
            data_df=self.train_df,
            int2sample=self.train_int2sample,
            sample2cells=self.train_sample2cells,
            sample_size=self.sample_size,
        )

        self.val_dataset = None
        if self.val_df is not None:
            gp = self.val_df.groupby(self.sampleID)
            self.val_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
            self.val_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
            #self.val_df["label_int"] = self.val_df[self.label_name_column].map(
            #    self.label2int
            #)
            self.val_dataset = self.dataset_cls(
                data_df=self.val_df,
                int2sample=self.val_int2sample,
                sample2cells=self.val_sample2cells,
                sample_size=self.sample_size,
            )

        self.test_dataset = None
        if self.test_df is not None:
            gp = self.test_df.groupby(self.sampleID)
            self.test_int2sample = {i: x for i, x in enumerate(gp.groups.keys())}
            self.test_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
            #self.test_df["label_int"] = self.test_df[self.label_name_column].map(
            #    self.label2int
            #)
            self.test_dataset = self.dataset_cls(
                data_df=self.test_df,
                int2sample=self.test_int2sample,
                sample2cells=self.test_sample2cells,
                sample_size=self.sample_size,
            )

        log.info("Datasets Created") 
    
    def align_gene_space(self):
        # gene space needs be aligned to the given gene order
        with open(self.gene_order, "r") as fh:
            self.gene_order = [line.strip() for line in fh]
        if self.gene_column == "genes":
            genes = self.gene_tdb.df[:]["genes"].tolist()
        elif self.gene_column == "cellarr_gene_index":
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
                log.info(f"Gene not found: {x}")
                pass
        self.n_genes = len(self.gene_indices)  # used when creating training model


    def divide_data(self):
        """
        Divides the data into training, validation and test sets.
        """
        self.val_df = None
        self.test_df = None

        if self.val_studies is not None:
            val_studies_col, val_studies_val = self.val_studies
            # split out validation studies
            self.val_df = self.data_df[
                self.data_df[val_studies_col].isin(val_studies_val)
            ].copy()
            self.train_df = self.data_df[
                ~self.data_df[val_studies_col].isin(val_studies_val)
            ].copy()
            
            if self.remove_new_val_labels: # Pure classification so we remove the classes not in the training data.
                log.info("Removing classes not in the training data from the validation data.")
                self.val_df = self.val_df[
                    self.val_df[self.label_name_column].isin(
                        self.train_df[self.label_name_column].unique()
                    )
                ].copy()
        else:
            self.train_df = self.data_df
        
        if self.test_studies is not None:
            test_studies_col, test_studies_val = self.test_studies
            self.test_df = self.data_df[
                self.data_df[test_studies_col].isin(test_studies_val)
            ].copy()
            self.train_df = self.train_df[
                ~self.train_df[test_studies_col].isin(test_studies_val)
            ].copy()
            # limit test celltypes to those in the training dat
            
            if self.remove_new_val_labels: # Pure classification so we remove the classes not in the training data.
                log.info("Removing classes not in the training data from the test data.")
                self.test_df = self.test_df[
                    self.test_df[self.label_name_column].isin(
                        self.train_df[self.label_name_column].unique()
                    )
            ].copy()
                

        ###TODO:  Just for consistency with the initial PaSCient repo.
        
        if self.dataset_name == "pascient_multilabel":

            onto = import_cell_ontology()
            id2name = get_id_mapper(onto)
            self.train_df["cellTypeName"] = self.data_df["celltype_id"].map(id2name)
            self.val_df["cellTypeName"] = self.data_df["celltype_id"].map(id2name)
            self.test_df["cellTypeName"] = self.data_df["celltype_id"].map(id2name)
            self.data_no_val = self.data_df.loc[~self.data_df["study"].isin(self.val_df["study"].unique())].copy()
            self.data_no_val["cellTypeName"] = self.data_no_val["celltype_id"].map(id2name)

            self.val_df = self.val_df[
                self.val_df["cellTypeName"].isin(
                    self.data_no_val["cellTypeName"].unique()
                )
            ]
            self.test_df = self.test_df[
                self.test_df["cellTypeName"].isin(
                    self.train_df["cellTypeName"].unique()
                )
            ]
            del self.data_no_val

        log.info(f"Training data size: {self.train_df.shape}")
        if self.val_df is not None:
            log.info(f"Validation data size: {self.val_df.shape}")
        if self.test_df is not None:
            log.info(f"Test data size: {self.test_df.shape}")

    def query_db(self):
        """" Query tileDB.

        Sets the data_df attribute to a pandas DataFrame containing the metadata.

        Also handles cached min-version of the data for faster loading in development.

        """

        if self.cached_db is not None: #Retrieve from cache if it exists.
            if path_exists(os.path.join(self.cached_db,"data_df.csv")):
                log.info("Fetching cached data_df...")
                self.data_df = pd.read_csv(os.path.join(self.cached_db,"data_df.csv"), index_col = "index")
                log.info("Done")
                return

        # limit to cells with counts and labels
        log.info("Querying tileDB...")

        if self.dataset_name == "pascient_multilabel":
            query_condition = f"total_counts > 0"
        elif self.dataset_name == "pascient_binary":
            query_condition = f"tissue == 'blood'"
        elif (self.dataset_name == "AD") or (self.dataset_name == "scimilarity"):
            query_condition = (
                f"{self.label_id_column} not in {self.nan_strings} and total_counts > 0"
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not recognized.")
        
        columns = [self.study_column, self.sample_column, self.label_id_column , "total_counts"] + self.extra_columns

        self.data_df = query_tiledb_df(
            self.cell_tdb,
            query_condition,
            attrs=columns,
        )

        #Additional conditions
        if self.dataset_name == "pascient_multilabel":
            diseases = ['COVID-19', "Crohn's disease", "Alzheimer's disease", 'lung adenocarcinoma', 'multiple sclerosis', 'melanoma', 'multiple myeloma', 'B-cell acute lymphoblastic leukemia', 'healthy']
            self.data_df = self.data_df[self.data_df["disease"].isin(diseases)]
        elif self.dataset_name == "pascient_binary":
            diseases = ['healthy', 'COVID-19']
            self.data_df = self.data_df[self.data_df["disease"].isin(diseases)]

        log.info("Done")

        if self.cached_db is not None:
            if not path_exists(os.path.join(self.cached_db,"data_df.csv")):
                random.seed(42)
                log.info("Saving cache data_df...")
                self.data_df = pd.concat(random.sample([df for _,df in self.data_df.groupby(["study"])], 5 ))
                self.data_df.to_csv(os.path.join(self.cached_db,"data_df.csv"), index=True, index_label = "index")
                log.info("Done")
        
    def collate(self,batch):
        """Collate tensors.

        Parameters
        ----------
        batch:
            Batch to collate.

        Returns
        -------
        tuple
            A Tuple[torch.Tensor, df] containing the normalized gene expression and the metadata of each cell.
        """
        
        sample_sizes = [ len(b) for i,b in enumerate(batch)] 
        sample_id = np.repeat(np.arange(len(batch)),sample_sizes)

        df = pd.concat(batch)
        df["sample_id_batch"] = sample_id
        cell_idx = df.index.tolist()
        
        if self.cached_db is not None: #Retrieve from cache if it exists.
            counts = self.cached_counts

        else: #Retrieve from tileDB
            matrix_tdb = tiledb.open(self.matrix_tdb_uri, "r", config = self.tdb_cfg)
            results = matrix_tdb.multi_index[cell_idx, :]
            matrix_tdb.close()

            if "vals" in results.keys():
                counts = coo_matrix(
                    (results["vals"], (results["x"], results["y"])),
                    shape=self.matrix_shape,
                ).tocsr()
            else:
                counts = coo_matrix(
                    (results["data"], (results["cell_index"], results["gene_index"])),
                    shape=self.matrix_shape,
                ).tocsr()
                 
        counts = counts[cell_idx, :]
        counts = counts[:, self.gene_indices]

        X = counts.astype(np.float32)

        X = torch.Tensor(X.toarray())
        if self.sparse:
            X = X.to_sparse_csr()

        batch_out = (
            X,
            df
        )

        batch_out = self.output_map(batch_out)

        return batch_out
    
    def train_dataloader(self) -> DataLoader:
        """Load the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the training dataset.
        """

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            batch_sampler=self.train_sampler,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
            worker_init_fn = worker_init_fn
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

        if self.multiprocessing_context == "spawn":
            # We can avoid too much memory usage by lowering the number of workers for validation.
            # Spawn creates separated processes for each worker.
            num_workers = self.num_workers // 4 + 1
        else:
            num_workers = self.num_workers
        
        return DataLoader(
            self.val_dataset,
            num_workers=num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            batch_sampler=self.val_sampler,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
            worker_init_fn = worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """

        if self.multiprocessing_context == "spawn":
            # We can avoid too much memory usage by lowering the number of workers for validation.
            # Spawn creates separated processes for each worker.
            num_workers = self.num_workers // 4 + 1
        else:
            num_workers = self.num_workers

        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            batch_sampler=self.test_sampler,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
            worker_init_fn = worker_init_fn
        ) 

if __name__ == "__main__":
    
    #Testing
    from pascient.data.data_samplers import BaseSampler, TissueStudySampler
    from pascient.data.batch_mappers import BatchMapper


    sampler = PartialClass(ref_class = "pascient.data.data_samplers.TissueStudySampler")

    dm2 = PatientCellsDataModule(
    cell_tdb_uri = "/projects/global/gred/resbioai/CeLLM/tiledb/cellarr_scimilarity_complete/cell_metadata",
        gene_tdb_uri =  "/projects/global/gred/resbioai/CeLLM/tiledb/cellarr_scimilarity_complete/gene_annotation",
        matrix_tdb_uri = "/projects/global/gred/resbioai/CeLLM/tiledb/cellarr_scimilarity_complete/counts",
        gene_order = "/gstore/data/omni/scdb/cleaned_h5ads/gene_order.tsv",
        label_id_column = "celltype_id",
        study_column = "study",
        sample_column =  "cellarr_sample",
        extra_columns= ["tissue", "disease"],
        batch_size =  4,
        sample_size = 200,
        num_workers = 0,
        sampler_cls = sampler,
        output_map = BatchMapper(pad_size = 200, sample_labels = ["disease", "tissue"], cell_labels = ["cellarr_sample"]),
        cached_db = "/projects/global/gred/resbioai/CeLLM/tiledb/small_db/",
        overwrite_cache= False)

    dl2 = dm2.train_dataloader()
    ds2 = dm2.train_dataset

    import time
    start_time = time.time()
    for i,b in enumerate(dl2):
        print(i)
        a = 0
        breakpoint()
    print(time.time()-start_time)