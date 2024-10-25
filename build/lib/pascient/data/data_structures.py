from dataclasses import dataclass

import torch


@dataclass
class CellSample:
    x: torch.Tensor
    pad: torch.Tensor

    @staticmethod
    def collate(batch):
        device = batch[0].x.device
        collated = {}
        keys = batch[0].__dict__.keys()
        for key in keys:
            attribute_list = [getattr(b, key) for b in batch]
            collated[key] = torch.stack(attribute_list).to(device=device)
        return CellSample(**collated)


@dataclass
class CellDataMasked(CellSample):
    mask: torch.Tensor
