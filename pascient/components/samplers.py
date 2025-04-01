import numpy as np


class Sampler():
    """
    Base class for samplers
    """

    def __init__(self):
        pass

    def sample(self, n: int) -> np.ndarray:
        """
        Sample n values from the distribution
        """
        raise NotImplementedError
    

class GaussianSampler(Sampler):
    """
    Sampler that samples from a Gaussian distribution
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(loc=self.mean, scale=self.std, size=n)
    
class UniformSampler(Sampler):
    """
    Sampler that samples from a Uniform distribution
    """

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(low=self.low, high=self.high, size=n)