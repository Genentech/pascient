from typing import List

import torch
from torch import nn

from cellm.components.misc import LinearModule, MLP, GMLP


class GeneToCellLinear(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LinearModule(input_dim=n_genes, output_dim=latent_dim)
        self.n_genes = n_genes
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GeneToCellMLP(nn.Module):
    def __init__(
            self,
            n_genes: int,
            latent_dim: int = 128,
            hidden_dim: List[int] = (1024, 1024),
            dropout: float = 0.,
            residual: bool = False,
            batchnorm = False,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim=n_genes, output_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout,
                           residual=residual, batchnorm = batchnorm)

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)


class GeneToCellGMLP(nn.Module):
    def __init__(
            self,
            n_genes: int,
            seq_len: int = 1500,
            latent_dim: int = 128,
            hidden_dim: List[int] = (1024, 1024),
            dropout: float = 0.,
            residual: bool = False,
            batchnorm = False,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.encoder = GMLP(input_dim=n_genes, output_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout, seq_len = seq_len,
                           residual=residual, batchnorm = batchnorm)

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)