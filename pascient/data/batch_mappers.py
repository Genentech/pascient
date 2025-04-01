import os
from typing import List, Optional, Tuple, Callable, Union
import torch
import numpy as np
from pascient.data.data_structures import SampleBatch, CellMetaData

from pascient.components.ontology import build_ancestry_matrix

import pickle

import logging
log = logging.getLogger(__name__)


class BatchMapper:
    """ Base class that takes the output of the collate and turns it into a more torch-useful format.

    Parameters
    ----------
    pad_size:
        Size of the padding. (Max number of cells per sample)
    sample_labels:
        List of list (col,type) to extract from the metadata to assign to each sample.
        For each tuple, the col is the name of the column in the metadata dataframe. 
        The type can be either 'categorical' or 'continuous'. If categorical, we assign an integer to each unique value in the column. If continuous, we directly use the values from the column and normalize them.
        E.g. [['sample_id', 'categorical'], ['age', 'continuous']]
    cell_labels:
        Same as sample_labels but this is for metadata at the cell level.
    return_index:
        Wether to return the index of each cell in the original dataframe.

    Returns
    -------
    tuple
        A Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] containing the padded tensor, the padding mask, the sample labels and the cell labels.
        Size of padded tensor = (n_samples, pad_size, n_genes)
        Size of padding mask = (n_samples, pad_size)
        Size of sample labels = (n_samples, len(sample_labels))
        Size of cell labels = (n_samples, pad_size, len(cell_labels))
    """

    def __init__(self, pad_size, sample_labels, cell_labels, return_ontology = True, return_index = False):
        self.pad_size = pad_size
        self.sample_labels = sample_labels
        self.cell_labels = cell_labels
        self.return_ontology = return_ontology
        self.return_index = return_index

    def set_dict(self, sample_labels_dict):
        """
        Set the dictionaries for the sample and cell labels (to convert them to integers)

        Parameters
        ----------
        sample_labels_dict:
            Dictionary containing the mappings of the sample and cell labels to integers. All fields in sample_labels and cell_labels should be present.
            e.g. sample_labels_dict[self.sample_labels[0]] = {'label1': 0, 'label2': 1, ...}
        """
        for label in self.sample_labels:
            assert label[0] in sample_labels_dict.keys(), f"Label {label} not found in sample_labels_dict - will not be able to map to integer"
        for label in self.cell_labels:
            assert label[0] in sample_labels_dict.keys(), f"Label {label} not found in sample_labels_dict - will not be able to map to integer"
        self.labels2int = sample_labels_dict

    def init_mapping(self,df, onto = None):
        """
        Creates a dictionary for each field in the sample_labels and cell_labels.
        Each key in this dictionary is a variable that will be fed in the batch.
        The values of the dictionary are the values in the initial metadata dataframe mapped with a dictionary.
        Mapping:
        - If the field is categorical, we map each unique value to an integer.
        - If the field is continuous, we map the value to a float.
        """
        # Sets the mapping between the fields in the dataframe and integers

        
        mapper_dict = {}
        for label_, metadata_type in self.cell_labels + self.sample_labels:
            if metadata_type == 'categorical':
                if df[label_].dtype == float:
                    dict_ = {sample: int(sample) for  sample in df[label_].unique()}
                else:
                    dict_ = {sample:i for i, sample in enumerate(df[label_].unique())}
                mapper_dict[label_] = dict_
            elif metadata_type == "continuous":
                #Normalizing the continuous variable
                col_continuous = df[label_].astype(float)
                col_mean = col_continuous.mean()
                col_std = col_continuous.std()
                #Creating the dictionary to map each value to the normalized equivalent.
                dict_ = {sample: (float(sample)-col_mean)/col_std for _, sample in enumerate(df[label_].unique())}
                mapper_dict[label_] = dict_

        self.set_dict(mapper_dict)

        self.create_ancestry_matrix(onto)

        return
    
    def create_ancestry_matrix(self, onto):
        #Computing ancestry similarity matrix (for relational learning - e.g. X-constrastive)
        if "celltype_id" in self.cell_labels:
            if self.return_ontology:
                log.info("Cell type id is provided. Building ancestry matrix.")
                self.ancestry_matrix = torch.LongTensor(build_ancestry_matrix(onto, self.labels2int["celltype_id"]))
            else:
                log.info("Cell type id is not provided. Will not build ancestry matrix.")
                self.ancestry_matrix = None
        else:
            self.ancestry_matrix = None
        return

    def set_augmentations(self, augmentations: dict): 
        """
        Set the augmentations to apply to the data.

        Parameters
        ----------
        augmentations:
            List of functions to apply to the data.
        """
        self.augmentations = augmentations

    def set_normalization(self, lognorm: bool, target_sum: Union[int,float]):
        self.lognorm = lognorm
        self.target_sum = target_sum

    def process_augmentations(self, batch: SampleBatch):
        for name, augmentation in self.augmentations.items():
           batch = augmentation(batch)
        return batch
    
    def normalize(self,  batch: SampleBatch)->SampleBatch:
        #normalization (log1p)

        if self.lognorm:

            X = batch.x
            pad_mask = batch.padded_mask

            counts_per_cell = X.sum(axis=-1) +1e-8
            counts_per_cell = counts_per_cell / self.target_sum

            counts_per_cell[~pad_mask] = 1
 
            X_padded_norm = X / counts_per_cell[..., None]

            X_padded_out = X_padded_norm.log1p()

            batch.x = X_padded_out
            return batch
        
        else: #do nothing
            return batch
        

    def __call__(self,batch:Tuple)-> Tuple[torch.Tensor,torch.Tensor,dict, dict]:

        """
        Take a batch and return a padded tensor, a padding mask, sample labels and cell labels.
        
        Parameters
        ----------
        batch:
            Tuple containing the data and metadata of the batch. 
            The data should be a torch.Tensor of size (n_cells, n_genes) and the metadata should be a pandas DataFrame with the metadata for each cell.

        Returns
        -------
        tuple
            A Tuple[torch.Tensor, torch.Tensor, dict, dict] containing the padded tensor, the padding mask, the sample labels dict and the cell labels dict.
            Size of padded tensor = (n_samples, n_views, pad_size, n_genes)
            Size of padding mask = (n_samples, n_views, pad_size)
            Size of sample labels = (n_samples, len(sample_labels))
            Size of cell labels = (n_samples, n_views, pad_size, len(cell_labels))
        """

        X, df = batch
        X_padded_list = []
        pad_list = []

        n_samples = df.sample_id_batch.max()+1
        n_views = df.view_id.max()+1 

        types_dict = {"continuous": float, "categorical": int} #mapping each label type to the right tensor type.

        sample_labels_dict = {label: torch.zeros(n_samples, dtype = types_dict[label_type]) for label, label_type in self.sample_labels}
        cell_labels_dict = { label: torch.zeros((n_samples, n_views, self.pad_size), dtype = types_dict[label_type]) for label, label_type in self.cell_labels}
        
        if self.return_index:
            cell_labels_dict["index"] = torch.zeros((n_samples, n_views, self.pad_size), dtype = int)

        for sample_id in range(n_samples):

            X_padded_views = []
            pad_views = []

            for view_id in range(n_views):

                df_mask = (df.sample_id_batch == sample_id) & (df.view_id == view_id)
                
                X_ = X[df_mask.values]
                X_padded = torch.zeros(self.pad_size, X_.shape[1])
                pad = torch.zeros(self.pad_size, dtype=torch.bool)
                
                X_padded[:X_.shape[0], :] = X_
                pad[:X_.shape[0]] = True


                try:
                    for sample_label, sample_label_type in self.sample_labels:
                        label = df.loc[df_mask, sample_label].iloc[0]
                        sample_labels_dict[sample_label][sample_id] = self.labels2int[sample_label][label]
                except:
                    breakpoint()
                for cell_label, cell_label_type in self.cell_labels:
                    label = torch.Tensor(df.loc[df_mask, cell_label].map(self.labels2int[cell_label]).values)
                    cell_labels_dict[cell_label][sample_id, view_id, :X_.shape[0]] = label

                if self.return_index:
                    cell_labels_dict["index"][sample_id, view_id, :X_.shape[0]] = torch.Tensor(df.index[df_mask].tolist())

                X_padded_views.append(X_padded)
                pad_views.append(pad)

            pad_list.append(torch.stack(pad_views))
            X_padded_list.append(torch.stack(X_padded_views))
 
        X_padded = torch.stack(X_padded_list)
        pad = torch.stack(pad_list)

        view_names = [f'view_{i}' for i in range(n_views)]

        cell_metadata = CellMetaData(cell_level_labels = cell_labels_dict, ancestry_matrix = self.ancestry_matrix)

        batch = SampleBatch( x = X_padded,
                            padded_mask = pad,
                     sample_metadata = sample_labels_dict,
                     cell_metadata = cell_metadata,
                     view_names = view_names)
        
        # Check if X_padded has nan
        augmented_batch = self.process_augmentations(batch)
        
        normalized_augmented_batch = self.normalize(augmented_batch)

        if augmented_batch.x.isnan().any():
            breakpoint()

        return normalized_augmented_batch
    
class PaSCientDiseaseMapper(BatchMapper):
    """
    This is a Mapper class for the PaSCient model that uses a fixed dictionary for the disease labels.
    """
    def __init__(self, pad_size, sample_labels, cell_labels, disease_dict, return_ontology = False, return_index = False):
        """
        disease_dict: Path to the dictionary containing the mapping of the disease labels to integers.
        """
        super().__init__(pad_size, sample_labels, cell_labels, return_ontology = return_ontology, return_index = return_index)

        with open(disease_dict, 'rb') as handle:
            self.disease_dict = pickle.load(handle)
        
    def init_mapping(self,df, onto = None):
        """
        Creates a dictionary for each field in the sample_labels and cell_labels.
        Each key in this dictionary is a variable that will be fed in the batch.
        The values of the dictionary are the values in the initial metadata dataframe mapped with a dictionary.
        Mapping:
        - If the field is categorical, we map each unique value to an integer.
        - If the field is continuous, we map the value to a float.
        """
        # Sets the mapping between the fields in the dataframe and integers


        mapper_dict = {}
        for label_, metadata_type in self.cell_labels + self.sample_labels:
            if label_ == "disease":
                dict_ = self.disease_dict
                mapper_dict[label_] = dict_
            else:
                if metadata_type == 'categorical':
                    if df[label_].dtype == float:
                        dict_ = {sample: int(sample) for  sample in df[label_].unique()}
                    else:
                        dict_ = {sample:i for i, sample in enumerate(df[label_].unique())}
                    mapper_dict[label_] = dict_
                elif metadata_type == "continuous":
                    #Normalizing the continuous variable
                    col_continuous = df[label_].astype(float)
                    col_mean = col_continuous.mean()
                    col_std = col_continuous.std()
                    #Creating the dictionary to map each value to the normalized equivalent.
                    dict_ = {sample: (float(sample)-col_mean)/col_std for _, sample in enumerate(df[label_].unique())}
                    mapper_dict[label_] = dict_

        self.set_dict(mapper_dict)
        
        self.create_ancestry_matrix(onto)


