"""
PROTOTYPE ONLY MODULE
This module includes classes/functions to load data from anndata directly for prototyping purposes.
Check data.data_scimilarity.py for the actual data functions.
"""

from random import Random

import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cellm.data.data_structures import CellSample


class CellSampleDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            adata,
            sample_label: str = 'study-sample',
            pad_size=20,
            masking_strategy=None
    ):
        self.adata = adata
        self.sample_label = sample_label

        self.sample_list = self.adata.obs[sample_label].unique()
        self.pad_size = pad_size
        self.masking_strategy = masking_strategy

        self.precomputed = False
        self.precomputed_list = []

    def __len__(self) -> int:
        return self.sample_list.shape[0]

    def __getitem__(self, idx: int):
        if self.precomputed:
            return self.precomputed_list[idx]

        sample_id = self.sample_list[idx]
        x_sample_id = torch.tensor(self.adata[self.adata.obs[self.sample_label] == sample_id].X, dtype=torch.float32)

        pad = torch.tensor([1] * x_sample_id.shape[0] + [0] * (self.pad_size - x_sample_id.shape[0]), dtype=torch.bool)
        x_sample_id_padded = torch.zeros(self.pad_size, x_sample_id.shape[1])
        x_sample_id_padded[:x_sample_id.shape[0], :] = x_sample_id

        if self.masking_strategy is None:
            cell_sample = CellSample(x=x_sample_id_padded, pad=pad)
        else:
            _, mask = self.masking_strategy(x_sample_id_padded)
            cell_sample = CellSample(x=x_sample_id_padded, pad=pad)
            cell_sample.mask = mask

        return cell_sample

    def precompute(self):
        self.precomputed_list = [self.__getitem__(i) for i in tqdm(range(len(self)))]
        self.precomputed = True


class CellSampleDataModule(L.LightningDataModule):
    def __init__(
            self,
            adata,
            sample_label: str = 'study-sample',
            pad_size=20,
            seed=0,
            split_sizes=(0.8, 0.1, 0.1),
            val_test_masking=None,
            num_workers_train=8,
            num_workers_test=8,
            train_batch_size=32,
            test_batch_size=64,
    ):
        self.adata = adata

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.adata_test = None
        self.adata_val = None
        self.adata_train = None

        self.sample_label = sample_label

        self.sample_list = self.adata.obs[sample_label].unique()
        self.pad_size = pad_size
        self.seed = seed
        self.split_sizes = split_sizes
        self.val_test_masking = val_test_masking
        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.split_data()

    def split_data(self):
        random = Random(self.seed)
        indices = list(range(self.sample_list.shape[0]))
        random.shuffle(indices)

        sizes = self.split_sizes

        train_size = int(sizes[0] * len(self.sample_list))
        train_val_size = int((sizes[0] + sizes[1]) * len(self.sample_list))

        train_samples = self.sample_list[:train_size]
        val_samples = self.sample_list[train_size:train_val_size]
        test_samples = self.sample_list[train_val_size:]

        self.adata_train = self.adata[self.adata.obs[self.sample_label].isin(train_samples)]
        self.adata_val = self.adata[self.adata.obs[self.sample_label].isin(val_samples)]
        self.adata_test = self.adata[self.adata.obs[self.sample_label].isin(test_samples)]

        self.train_dataset = CellSampleDataset(self.adata_train, sample_label=self.sample_label, pad_size=self.pad_size)
        self.val_dataset = CellSampleDataset(self.adata_val, sample_label=self.sample_label, pad_size=self.pad_size,
                                             masking_strategy=self.val_test_masking)
        self.test_dataset = CellSampleDataset(self.adata_test, sample_label=self.sample_label, pad_size=self.pad_size,
                                              masking_strategy=self.val_test_masking)

        self.val_dataset.precompute()
        self.test_dataset.precompute()

        self.adata_train, self.adata_val, self.adata_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers_train)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers_test)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers_test)
