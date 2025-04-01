from typing import List

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from pascient.components.misc import MLP
from pascient.utils.torch_utils import get_L_dim


class CellToCellPytorchTransformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            n_heads: int,
            dim_feedforward: int = None,
            dropout: float = 0.,
            num_layers: int = 3,
            single_cell_only: bool = False,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = input_dim * 2

        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.network = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.single_cell_only = single_cell_only

    def back_forward(self, batch, src_key_padding_mask=None, src_mask=None) -> torch.Tensor:
        if self.single_cell_only:  # attention only to the same cell - deprecated
            breakpoint()
            batch_L = get_L_dim(batch)
            attention_mask = torch.ones((batch_L, batch_L), dtype=torch.bool, device=batch.device)
            torch.diagonal(attention_mask, 0).fill_(False)
            return self.network(batch, mask=attention_mask)
    
        o = self.network(batch, src_key_padding_mask=src_key_padding_mask, mask=src_mask)
        return o
    
    def forward(self, x, padding_mask):

        x_ = x.reshape(-1, x.shape[2], x.shape[3])
        padding_mask_ = padding_mask.reshape(-1, padding_mask.shape[2])

        x_out = self.back_forward(x_, src_key_padding_mask= ~padding_mask_)
        
        return x_out.reshape(x.shape[:-1]+(x_out.shape[-1],))


class CellToCellMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 128,
            hidden_dim: List[int] = (1024, 1024),
            dropout: float = 0.,
            residual: bool = False,
    ):
        super().__init__()
        self.encoder = MLP(input_dim=input_dim, output_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout,
                           residual=residual)

    def forward(self, x, **kwargs) -> torch.Tensor:
        return self.encoder(x)
    
class CellToCellIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs) -> torch.Tensor:
        return x
