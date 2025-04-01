import torch
from pascient.data.data_structures import SampleBatch
from typing import Tuple

class Masking:
    """
    Base class for masking. Subclasses should implement `compute_mask`.
    :param mask_token:
    :param seed:
    :param device:
    """
    def __init__(self, views, mask_token=0, seed=None, device=None):
        self.generator = None
        if seed is not None:
            assert device is not None, "A device is needed if seed is provided"
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)

        self.mask_token = mask_token
        self.views = views # views for which to compute a mask version

    def compute_mask(self, batch: SampleBatch):
        raise NotImplementedError

    def __call__(self, batch: SampleBatch)->SampleBatch:

        self.compute_mask(batch)

        return self.apply_mask(batch)

    def apply_mask(self, batch:SampleBatch)->SampleBatch:
        
        view_names = batch.view_names
        x = batch.x

        view_idxs = [view_names.index(view) for view in self.views]

        if batch.dropout_mask is None:
            raise ValueError("No dropout mask found in the batch. Please run compute_mask first.")
        
        dropout_mask = batch.dropout_mask.to(dtype=torch.float)

        masked_batch = x[:,view_idxs] * dropout_mask + (1 - dropout_mask) * self.mask_token

        updated_view_names = view_names + [f"mask_from_{view}" for view in self.views]

        updated_padded_mask = torch.cat([batch.padded_mask, batch.padded_mask[:,view_idxs]],1)

        batch.padded_mask = updated_padded_mask
        batch.x = torch.cat((x, masked_batch), 1)
        batch.view_names = updated_view_names

        #UPDATE THE CELL METADATA
        cell_metadata = batch.cell_metadata
        for k in cell_metadata.cell_level_labels.keys(): #concatenate the cell level metadata from the original view to augment the view
            augmented_cell_labels = cell_metadata.cell_level_labels[k][:,view_idxs]
            cell_metadata.cell_level_labels[k] = torch.cat([cell_metadata.cell_level_labels[k], augmented_cell_labels],axis = 1)
        batch.cell_metadata = cell_metadata

        return batch
    
    def compare_reconstructions(self, true_batch: SampleBatch, predictions: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the predictions and the true batch, returns the tensors to be compared for loss computation.
        :param true_batch: Original batch of data given to the model
        :param predictions: Torch Tensor as the output of the model. Shape is [bs, n_views, n_cells, n_genes] This should be the same dimensions of batch.x
        :return: 
            Tensors to be compared, selected only where a) the gene/cell combination have been masked and b) the cells are not padded.
            true_x: The true values to be compared. Shape is [m_cells, n_genes]
            pred_x: The predicted values to be compared. Shape is [m_cells, n_genes]
        """

        view_names = true_batch.view_names
        pad_mask = true_batch.padded_mask
        dropout_mask = true_batch.dropout_mask

        # select only the views we want to consider.
        gt_views_idx = [view_names.index(view) for view in self.views] # for the ground truth
        rec_views_idx = [view_names.index(f"mask_from_{view}") for view in self.views] # for the reconstruction
        
        # select only the cells that are non padded and masked.
        loss_mask = pad_mask[:,gt_views_idx][...,None] * ~(dropout_mask.bool())

        true_x = true_batch.x[:,gt_views_idx][loss_mask]
        pred_x = predictions[:,rec_views_idx][loss_mask]

        return true_x, pred_x


class DummyMasking:
    """
    Dummy masking that does not mask anything.
    """
    def __init__(self, **kwargs):
       return 

    def __call__(self, batch: SampleBatch)->SampleBatch:
        return batch

class MaskRandomGenes(Masking):
    """
    Mask genes in each cell randomly, according to probability p (per-cell).
    :param batch:
    :param mask_p: masking probability
    :param mask_token:
    :return masked_batch, mask. True means kept, False means masked.
    """

    def __init__(self, mask_p, views, mask_token=0, seed=None, device=None):
        super().__init__( views = views, mask_token=mask_token, seed=seed, device=device)
        self.mask_p = mask_p
        self.keep_prob = 1 - self.mask_p  # probability of keeping

    def compute_mask(self, batch: SampleBatch):

        view_idxs = [batch.view_names.index(view) for view in self.views]
        x = batch.x

        masks = [torch.bernoulli(torch.full_like(x[:,view_idx], self.keep_prob), generator=self.generator) for view_idx in view_idxs]

        batch.dropout_mask = torch.stack(masks,1)
        return batch


class MaskRandomGenesInRandomCells(Masking):
    """
    For each cell, mask a fraction of genes with probability mask_p_cell. Genes are masked with probability mask_p_gene.
    :param batch:
    :param mask_p: masking probability
    :param mask_token:
    :return masked_batch, mask. True means kept, False means masked.
    """

    def __init__(self, mask_p_gene, mask_p_cell, mask_token=0, seed=None, device=None):
        super().__init__(mask_token=mask_token, seed=seed, device=device)
        self.mask_p_gene = mask_p_gene
        self.mask_p_cell = mask_p_cell
        self.keep_prob_gene = 1 - self.mask_p_gene
        self.keep_prob_cell = 1 - self.mask_p_cell

    def compute_mask(self, x):
        breakpoint() # this should be reimplemented such that it deals with sample views.
        cell_is_kept = torch.rand(x.shape[:-1]) < self.keep_prob_cell
        mask_genes = torch.bernoulli(torch.full_like(x, fill_value=self.keep_prob_gene),
                                     generator=self.generator)
        mask_genes[cell_is_kept] = 1
        return mask_genes

