import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np
from pascient.components.samplers import GaussianSampler, UniformSampler
from pascient.data.data_structures import SampleBatch, CellMetaData

class DataAugmentation(ABC):
    """ Base Class for Data Augmentation """
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, x: SampleBatch)-> SampleBatch:
        raise NotImplementedError
    

class DepthAugmentation(DataAugmentation):
    """
    Resamples the gene expression based on original distribution
    """

    def __init__(self, data_df: pd.DataFrame, n_views: int = 1, N_range: list[int] = [500,5000], **kwargs):
        """
        Initialize the DepthAugmentation.

        :param data_df: The complete data base
        :param n_views: The number of views to generate
        """

        self.n_views = n_views

        self.epsilon = 1e-9

        self.N_range = N_range
        
        #mu = data_df['total_counts'].mean()
        #std = data_df['total_counts'].std()

        #self.depth_sampler = GaussianSampler(mu, 0.1 * std)
        
        self.depth_sampler = UniformSampler(N_range[0], N_range[1])


    def __call__(self, batch: SampleBatch)->SampleBatch: #x: torch.Tensor, mask: torch.Tensor, cell_metadata:dict, view_names: list) -> torch.Tensor:
        """
        Augment the data by sampling new reads from empirical multinomial distribution

        :param batch : SampleBatch object with attributes
            x: tensor of shape (n_samples, n_views, n_cells, n_genes)
            mask: tensor of shape (n_samples, n_views, n_cells) (True if observed, False if missing)
            view_names: list of view names ( list of length n_views)
            cell_metadata: CellMetaData object with attributes
                cell_level_labels: dict of cell level labels (dict with each value being a numpy array of shape (n_samples, n_views, n_cells))
                ancestry_matrix: tensor of shape (M, M) with M the total number of cell types in the dataset.

        :return: New batch with augmented view.

        The new view is computed by computing the empirical multinomial distribution of the original view and sampling new reads from this distribution.
        The empirical distribution for a cell is computed as x_i / sim(x_i) where x_i is the original read count and sim(x_i) is the sum of all read counts in the cell.
        We include a small epsilon to ensure non-zero probability for each gene.

        New depths for each cell are sampled from a Uniform distribution with parameters self.N_range[0] and self.N_range[1].
        """

        view_names = batch.view_names
        x = batch.x
        mask = batch.padded_mask
        cell_metadata = batch.cell_metadata

        ref_view_name = "view_0" #always takes the first view as the reference view

        ref_view_idx = view_names.index(ref_view_name)
        ref_view = x[:,ref_view_idx]
        
        ref_mask = mask[:,ref_view_idx]

        new_depths = np.abs(self.depth_sampler.sample((x.shape[0], self.n_views)).astype(int))

        probs = (ref_view + self.epsilon) / (ref_view+self.epsilon).sum(2)[...,None] # compute resampling probabilities
 
        # ---- Re-sample to create new view -----
        augmented_tensor = []
        for i in range(self.n_views):
            augmented_view = []
            for j in range(ref_view.shape[0]):
                 samples = torch.multinomial(probs[j], new_depths[j,i], replacement = True)
                 new_counts = torch.stack([torch.bincount(samples[k_], minlength = ref_view.shape[-1]) for k_ in range(samples.shape[0])])
                 augmented_view.append(new_counts)
            augmented_view = torch.stack(augmented_view) * ref_mask[...,None]
            augmented_tensor.append(augmented_view)
        
        augmented_tensor = torch.stack(augmented_tensor,1)
        augmented_mask = torch.stack([ref_mask]*self.n_views, dim = 1)
        # ----------------

        # ---- Concatenate the original and augmented views ----
        x_out = torch.cat([x, augmented_tensor], 1) # Concatenate the original and augmented views
        mask_out = torch.cat([mask, augmented_mask], 1) # Concatenate the original and augmented mask

        view_names = view_names + [f'depth_augmentation_{i}_from_{ref_view_name}' for i in range(self.n_views)] # Update the view names

        for k in cell_metadata.cell_level_labels.keys(): #concatenate the cell level metadata from the original view to augment the view
            augmented_cell_labels = cell_metadata.cell_level_labels[k][:,[ref_view_idx]*self.n_views]
            cell_metadata.cell_level_labels[k] = torch.cat([cell_metadata.cell_level_labels[k], augmented_cell_labels],axis = 1)
        # ----------------

        # Update batch object
        batch.x = x_out
        batch.padded_mask = mask_out
        batch.view_names = view_names
        batch.cell_metadata = cell_metadata

        return batch