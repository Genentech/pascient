import torch

MASK_TOKEN = 0.


class Masking:
    """
    Base class for masking. Subclasses should implement `compute_mask`.
    :param mask_token:
    :param seed:
    :param device:
    """
    def __init__(self, mask_token=MASK_TOKEN, seed=None, device=None):
        self.generator = None
        if seed is not None:
            assert device is not None, "A device is needed if seed is provided"
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)

        self.mask_token = mask_token

    def compute_mask(self, x):
        raise NotImplementedError

    def __call__(self, x):
        mask = self.compute_mask(x)
        return self.apply_mask(x, mask)

    def apply_mask(self, x, mask):
        mask = mask.to(dtype=torch.float)
        masked_batch = x * mask + (1 - mask) * self.mask_token
        return masked_batch, mask.to(dtype=torch.bool)


class MaskRandomGenes(Masking):
    """
    Mask genes in each cell randomly, according to probability p (per-cell).
    :param batch:
    :param mask_p: masking probability
    :param mask_token:
    :return masked_batch, mask. True means kept, False means masked.
    """

    def __init__(self, mask_p, mask_token=MASK_TOKEN, seed=None, device=None):
        super().__init__(mask_token=mask_token, seed=seed, device=device)
        self.mask_p = mask_p
        self.keep_prob = 1 - self.mask_p  # probability of keeping

    def compute_mask(self, x):
        mask = torch.bernoulli(torch.full_like(x, self.keep_prob), generator=self.generator)
        return mask


class MaskRandomGenesInRandomCells(Masking):
    """
    For each cell, mask a fraction of genes with probability mask_p_cell. Genes are masked with probability mask_p_gene.
    :param batch:
    :param mask_p: masking probability
    :param mask_token:
    :return masked_batch, mask. True means kept, False means masked.
    """

    def __init__(self, mask_p_gene, mask_p_cell, mask_token=MASK_TOKEN, seed=None, device=None):
        super().__init__(mask_token=mask_token, seed=seed, device=device)
        self.mask_p_gene = mask_p_gene
        self.mask_p_cell = mask_p_cell
        self.keep_prob_gene = 1 - self.mask_p_gene
        self.keep_prob_cell = 1 - self.mask_p_cell

    def compute_mask(self, x):
        cell_is_kept = torch.rand(x.shape[:-1]) < self.keep_prob_cell
        mask_genes = torch.bernoulli(torch.full_like(x, fill_value=self.keep_prob_gene),
                                     generator=self.generator)
        mask_genes[cell_is_kept] = 1
        return mask_genes

