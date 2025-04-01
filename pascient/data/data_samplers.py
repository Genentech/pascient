from torch.utils.data import Sampler
import pandas as pd
import numpy.typing as npt
import numpy as np
import torch
import random 

from imblearn.over_sampling import RandomOverSampler ,SVMSMOTE

from typing import Iterator, List

import logging
log = logging.getLogger(__name__)

class BaseSampler(Sampler[int]):
    """
    Simplest sampler class for composition of studies in minibatch.
    """
    def __init__(
        self, data_df: pd.DataFrame, int2sample: dict, 
        bsz: int, shuffle: bool = True, **kwargs):
        """Constructor.
        
        Args:
            data_df: DataFrame with columns "study::::sample"
            int2sample: dictionary mapping integer to sample id
            bsz: batch size
            shuffle: whether to shuffle the samples across epochs
        """

        super().__init__()
 
        self.bsz = bsz
        self.data_df = data_df.copy()
        self.shuffle = shuffle
        self.int2sample = int2sample
        self.sample2int = {v: k for k, v in int2sample.items()}

        assert len(self.int2sample) == self.data_df["study::::sample"].nunique()
        

    def __len__(self) -> int:
        """
        number of batches in one epoch
        """
        return (len(self.int2sample) + self.bsz - 1) // self.bsz
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        returns a list of indices of each batch
        """
        batch_list = torch.LongTensor(list(self.int2sample.keys())) 
        if self.shuffle:
            batch_list = batch_list[torch.randperm(batch_list.shape[0])]
         
        for batch in torch.chunk(batch_list,len(self)):
            yield batch.tolist()


class TissueStudySampler(BaseSampler):
    
    """
    Sampler that creates minibatches where each sample is from the same tissue and same study when possible.

    Idea:

    - Split the samples by tissue.
    - For each tissue, 
        - randomize the order of studies
        - randomize order of samples within each study
        - split the samples into batches
    - Aggregate the batches from all tissues
    - Merge batches if they are too small (<1/2 batch size)
    - Sample batches at random

    """

    def __init__(
        self, data_df: pd.DataFrame, int2sample: dict, 
        bsz: int, shuffle: bool = True,  **kwargs):
        """Constructor.
        
        Args:
            data_df: DataFrame with columns "study::::sample"
            int2sample: dictionary mapping integer to sample id
            bsz: batch size
            shuffle: whether to shuffle the samples across epochs
        """

        super().__init__(data_df, int2sample, bsz, shuffle, **kwargs)

        self.data_df = data_df.copy()
        self.int2sample = int2sample
        self.sample2int = {v: k for k, v in int2sample.items()}

        assert len(self.int2sample) == self.data_df["study::::sample"].nunique() # just to make sure that the mapping is correct
        
        self.data_df_sample = data_df[["tissue","study","study::::sample"]].drop_duplicates()


    def split_data_in_batches(self)->List[pd.DataFrame]:
        """
        Split data into batches of size bsz
        """

        tissue_batches = [] 
        for _, tissue_df in self.data_df_sample.groupby(["tissue"]): #splitting the dataframe by tissue
            
            if self.shuffle:
                df_studies = [ df_.sample(frac=1) for _, df_ in tissue_df.groupby(["study"]) ] # randomize the order of samples within each study
                random.shuffle(df_studies) # randomize the order of studies
            else:
                df_studies = [ df_ for _, df_ in tissue_df.groupby(["study"]) ]

            tissue_df_rand = pd.concat(df_studies)

            n_batches = (len(tissue_df_rand) + self.bsz - 1) // self.bsz
            tissue_batches += [tissue_df_rand.iloc[i*self.bsz:(i+1)*self.bsz] for i in range(n_batches)] # split the samples into batches
        
        return tissue_batches
    
    def merge_batches(self, batches:List[pd.DataFrame])->List[pd.DataFrame]:
        """Merge batches if they are too small.

        Args:
            batches: list of DataFrames (batches)

        Returns:
            list of DataFrames where maximum 1 batch is smaller than bsz/2
        
        """
        batch_lengths = np.array([len(batch) for batch in batches])
        batches_to_merge = batch_lengths < 0.5 * self.bsz

        while batches_to_merge.sum() > 1:
            idx = np.where(batches_to_merge)[0][0]
            batches[idx] = pd.concat([batches[idx], batches[idx+1]])
            batches.pop(idx+1)
            batch_lengths = np.array([len(batch) for batch in batches])
            batches_to_merge = batch_lengths < 0.5 * self.bsz
        return batches


    def __iter__(self) -> Iterator[List[int]]:
        """
        returns a list of indices of each batch
        """
        tissue_batches = self.split_data_in_batches() # split data into batches
        batches = self.merge_batches(tissue_batches) # merge batches together if they are too small (<1/2 batch size)
        random_idx = np.random.permutation(len(batches)) if self.shuffle else np.arange(len(batches)) # randomize the order of batches

        for idx in random_idx:
            yield batches[idx]["study::::sample"].map(self.sample2int).tolist()


class OverSamplerPerAttribute(BaseSampler):
    def __init__(
        self, data_df: pd.DataFrame, int2sample: dict, 
        bsz: int, shuffle: bool = True, attributes: List[str] = None, sample_col:str = None, data_split:str = None, apply_on_splits:List[str] = ["train"], **kwargs):
        """Constructor.

        Oversamples the data per attributes. This inflates the data such that the number of samples per each attribute is the same.
        
        Args:
            data_df: DataFrame with columns "study::::sample"
            int2sample: dictionary mapping integer to sample id
            bsz: batch size
            shuffle: whether to shuffle the samples across epochs
            sample_col: the name of the column in the metadata df that corresponds to the sample id (should correspond to the keys in int2sample)
            attributes: list of attributes to oversample by.
            data_split: type of split this is initialized for (train, val, test)
            apply_on_splits: list of splits to apply the oversampling on - default : only on train.
        """

        super().__init__(data_df, int2sample, bsz, shuffle, **kwargs)

        self.bsz = bsz
        self.shuffle = shuffle
        #self.sample2int = {v: k for k, v in int2sample.items()}

        if not attributes:
            raise ValueError("Please provide attributes to oversample")
        self.attributes = attributes

        # Define mapping between new larger minibatch index and initial index (smaller)
        self.inflated_batch_idx_map = {i:i for i in range(len(int2sample))} # initialize the mapping to identity

        if data_split in apply_on_splits:    
            log.info(f"Oversampling data per attribute - initial data size : {len(int2sample)}")
            for attribute in self.attributes:
                log.info(f"Oversampling {attribute}")
                assert attribute in data_df.columns, f"{attribute} not found in data_df"
                gp_attribute = data_df.groupby([sample_col])[[sample_col,attribute]].head(1).set_index(sample_col)
                labels_list = []
                key_list = []
                for key , sample_ref in self.inflated_batch_idx_map.items(): #self.int2sample.items():
                    sample = int2sample[sample_ref]
                    labels_list.append(gp_attribute.loc[sample].item())
                    key_list.append(key)
                unique_, counts_ = np.unique(labels_list, return_counts=True)
                
                ros = RandomOverSampler(random_state=0)
                X_resampled, y_resampled = ros.fit_resample(np.array(key_list).reshape(-1, 1), labels_list)
                self.inflated_batch_idx_map = {i: self.inflated_batch_idx_map[v] for i, v in enumerate(X_resampled[:,0])}
                #int2sample_new = {k: self.int2sample[v] for k, v in enumerate(X_resampled[:,0])}
                #self.int2sample = int2sample_new.copy()
                log.info(f"Inflated data to {len(self.inflated_batch_idx_map)} samples") 

    def __len__(self) -> int:
        """
        number of batches in one epoch
        """
        return (len(self.inflated_batch_idx_map) + self.bsz - 1) // self.bsz
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        returns a list of indices of each batch
        """
        meta_batch_list = torch.LongTensor(list(self.inflated_batch_idx_map.keys())) 
        if self.shuffle:
            meta_batch_list = meta_batch_list[torch.randperm(meta_batch_list.shape[0])]
         
        for meta_batch in torch.chunk(meta_batch_list,len(self)):
            batch = [self.inflated_batch_idx_map[int(meta_idx)] for meta_idx in meta_batch]
            yield batch