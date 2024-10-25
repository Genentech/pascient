import torch
from torch import nn
from g_mlp_pytorch import gMLP
import torch.nn.functional as F
from typing import List


class LinearModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x) -> torch.Tensor:
        o = self.network(x)
        return o


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: List[int] = (1024, 1024),
        dropout: float = 0.,
        residual: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = output_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                if batchnorm:
                    self.network.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim[i]),
                            nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
                else:
                    self.network.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim[i]),
                            # nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
            else:  # hidden layers
                if batchnorm:
                    self.network.append(
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                            nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
                else:
                    self.network.append(
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                            # nn.BatchNorm1d(hidden_dim[i]),
                            nn.PReLU(),
                            # nn.Mish(),
                        )
                    )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], self.latent_dim))

    def forward(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

class GMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: List[int] = (1024, 1024),
        seq_len: int=1500,
        dropout: float = 0.,
        residual: bool = False,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = output_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                if batchnorm:
                    self.network.append(
                        nn.Sequential(
                            gMLP(
                                num_tokens = None,
                                dim = hidden_dim[i],
                                depth = 1,
                                seq_len = seq_len,
                                circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
                                act = nn.PReLU()              # activation for spatial gate (defaults to identity)
                            ),

                            nn.BatchNorm1d(hidden_dim[i]),
                            # nn.Mish(),
                        )
                    )
                else:
                    self.network.append(
                        nn.Sequential(
                            gMLP(
                                num_tokens = None,
                                dim = output_dim,
                                depth = 1,
                                seq_len = seq_len,
                                circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
                                act = nn.PReLU()              # activation for spatial gate (defaults to identity)
                            ),
                            # nn.BatchNorm1d(hidden_dim[i]),
                            # nn.Mish(),
                        )
                    )
            else:  # hidden layers
                if batchnorm:
                    self.network.append(
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            gMLP(
                                num_tokens = None,
                                dim = hidden_dim[i],
                                depth = 1,
                                seq_len = seq_len,
                                circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
                                act = nn.PReLU()              # activation for spatial gate (defaults to identity)
                            ),
                            nn.BatchNorm1d(hidden_dim[i]),
                            # nn.Mish(),
                        )
                    )
                else:
                    self.network.append(
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            gMLP(
                                num_tokens = None,
                                dim = hidden_dim[i],
                                depth = 1,
                                seq_len = seq_len,
                                circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
                                act = nn.PReLU()              # activation for spatial gate (defaults to identity)
                            ),
                            # nn.BatchNorm1d(hidden_dim[i]),
                            # nn.Mish(),
                        )
                    )
        # output layer
        self.network.append(gMLP(
                                num_tokens = None,
                                dim = hidden_dim[i],
                                depth = 1,
                                seq_len = seq_len,
                                circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
                                act = nn.PReLU()              # activation for spatial gate (defaults to identity)
                            ))

    def forward(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x
