import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch.nn as nn

class Aggregator(ABC):
    """
    Abstract base class for aggregators.
    """

    @abstractmethod
    def aggregate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate the data.

        :param data: The data to aggregate.
        :return: The aggregated data.
        """
        pass


class MeanAggregator(Aggregator):
    """
    Aggregator that computes the mean of the data.
    """

    def __init__(self, dim: int = 0):
        """
        Initialize the aggregator.

        :param dim: The dimension along which to compute the mean.
        """
        self.dim = dim

    def aggregate(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the data.

        :param data: The data to aggregate.
        :param mask: The mask to use for aggregation.
        :return: The aggregated data.
        """
        averaged_embeddings = (data * mask[...,None]).sum(dim= self.dim) / mask.sum(dim=self.dim)[...,None]
        return averaged_embeddings
    
class SumAggregator(Aggregator):
    """
    Aggregator that computes the sum of the data.
    """

    def __init__(self, dim: int = 0):
        """
        Initialize the aggregator.

        :param dim: The dimension along which to compute the mean.
        """
        self.dim = dim

    def aggregate(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the data.

        :param data: The data to aggregate.
        :param mask: The mask to use for aggregation.
        :return: The aggregated data.
        """
        breakpoint()
        summed_embeddings = (data * mask[...,None]).sum(dim= self.dim)
        return summed_embeddings
    
class NonLinearAttnAggregator(Aggregator,nn.Module):
    """
    Non Linear Attention Aggregator
    """

    def __init__(self, attention_model: nn.Module = None):
        """
        Initialize the aggregator.
        """
        super().__init__()
        self.attention_model = attention_model


    def aggregate(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the data.

        :param data: The data to aggregate.
        :param mask: The mask to use for aggregation.
        :return: The aggregated data.
        """
        attn_ = self.attention_model(data)
        sm = torch.nn.functional.softmax(attn_, dim=2)
        updated_data = sm * data
        summed_embeddings = updated_data.sum(dim = 2)
        return summed_embeddings
    
class LinearAttnAggregator(Aggregator):
    """
    Non Linear Attention Aggregator
    """

    def __init__(self):
        """
        Initialize the aggregator.
        """
        super().__init__()


    def aggregate(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the data.

        :param data: The data to aggregate.
        :param mask: The mask to use for aggregation.
        :return: The aggregated data.
        """
        sm = torch.nn.functional.softmax(data, dim=2)
        updated_data = sm * data
        summed_embeddings = updated_data.sum(dim = 2)
        return summed_embeddings