from abc import ABC, abstractmethod
import random


class Distribution(ABC):
    @abstractmethod
    def sample(self) -> float:
        ...

    @abstractmethod
    def mean(self) -> float:
        ...

    def __call__(self) -> float:
        return self.sample()
    

class NormalDistribution(Distribution):
    def __init__(self, mean: float, stddev: float):
        self._mean = mean
        self._stddev = stddev

    def sample(self) -> float:
        return random.gauss(self._mean, self._stddev)

    def mean(self) -> float:
        return self._mean
    
    def __repr__(self) -> str:
        return f"NormalDistribution(mean={self._mean}, stddev={self._stddev})"