from typing import Type

import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule

from pascient.model.base_lightning import BaseLightningModule


class BasicMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_hidden_layers: int = 0,
        activation_cls: Type[nn.Module] = nn.GELU,
        activation_out_cls: Type[nn.Module] = None,
    ):
        """
        if n_hidden_layers is -1, then the model will have no hidden layers and will be a simple linear model.

        activation_cls: the activation to apply for the hidden layers.
        activation_out_cls: activation to apply to the output layer. If None, no activation is applied.
        """
        super().__init__()
        self.in_features = input_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
        if n_hidden_layers > -1:
            self.layers.append(activation_cls())
            for i in range(n_hidden_layers):
                self.layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                self.layers.append(activation_cls())
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        if activation_out_cls is not None:
            self.layers.append(activation_out_cls())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x