from typing import List

import torch
from torch import nn

from cellm.components.misc import MLP


class CellToOutputMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 128,
            hidden_dim: List[int] = (1024, 1024),
            dropout: float = 0.,
            residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = output_dim
        self.encoder = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout,
                           residual=residual)

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)
