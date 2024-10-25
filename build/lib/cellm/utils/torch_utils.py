import torch
import torchmetrics
from torch import Tensor


def get_L_dim(x: torch.Tensor):
    """
    Returns the length dimension of the tensor x, assumes batch first.
    :param x:
    :return:
    """
    if x.ndim == 3:
        return x.shape[1]
    elif x.ndim == 2:
        return x.shape[0]
    else:
        raise ValueError


def combine_dims_except_last(x: torch.Tensor) -> torch.Tensor:
    """
    Combines all tensor dimensions except last one, return reshaped tensor.
    :param x:
    :return:
    """
    D = torch.prod(torch.tensor(x.size()[:-1]))
    x_reshaped = x.view(D, x.size()[-1])
    return x_reshaped


class BatchedR2(torchmetrics.R2Score):
    """
    Batched version of R2 score.
    """

    def __init__(self, num_outputs, columns=False):
        super().__init__(num_outputs=num_outputs)
        self.columns = columns

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds_reshaped = combine_dims_except_last(preds)
        gt_reshaped = combine_dims_except_last(target)
        if self.columns:
            preds_reshaped, gt_reshaped = preds_reshaped.T, gt_reshaped.T
        return super().update(preds_reshaped, gt_reshaped)
