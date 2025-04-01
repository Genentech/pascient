import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import functools
import pascient
import s3fs
from io import BytesIO

import os
from scipy.sparse import save_npz, load_npz

import logging
log = logging.getLogger(__name__)

class PartialClass:
    def __init__(self,ref_class, **kwargs):
        import sys
        class_name = ref_class.split(".")[-1]
        module_name = ".".join(ref_class.split(".")[:-1])
        cls = getattr(sys.modules[module_name],class_name)
        
        self.partial_cls = functools.partial(cls, **kwargs)

    def __call__(self, **kargs):
        return self.partial_cls(**kargs)


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
                self.network.append(
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        # nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
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


def path_exists(path:str):
    """"
    Check wether a path exists in S3 or local filesystem
    (if the first two characters are "s3" it is assumed to be an S3 path)
    """

    if path[:2] == "s3":
        log.info(f"Checking if {path} exists in S3")
        from s3fs.core import S3FileSystem
        fs = S3FileSystem()
        return fs.exists(path)
    else:
        import os
        return os.path.exists(path)
    
def save_sparse_matrix(path, matrix):
    """
    Save a sparse matrix to a file in S3 or local filesystem
    """
    
    if path[:2] == "s3":
        log.info(f"Saving matrix to S3 bucket at {path}...")
        s3 = s3fs.S3FileSystem()
        
        with s3.open(path, 'wb') as f:
            buffer = BytesIO()
            save_npz(buffer, matrix)
            buffer.seek(0)
            f.write(buffer.read())
    else:
        log.info(f"Saving matrix to local filesystem at {path}")
        save_npz(path, matrix)
    
    return

def load_sparse_matrix(path):
    """
    Load a sparse matrix from a file in S3 or local filesystem
    """
    if path[:2] == "s3":
        log.info(f"Loading matrix from S3 bucket at {path}")
        s3 = s3fs.S3FileSystem()
        f = s3.open(path, 'rb')
        return load_npz(f)
    else:
        log.info(f"Loading matrix from local filesystem at {path}")
        return load_npz(path)
    
def remove_file(path):
    """
    Remove a file from S3 or local filesystem
    """

    if path[:2] == "s3":
        log.info(f"Removing file from S3 bucket at {path}")
        s3 = s3fs.S3FileSystem()
        s3.rm(path)
    else:
        log.info(f"Removing file from local filesystem at {path}")
        os.remove(path)