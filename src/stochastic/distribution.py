from abc import ABC, abstractmethod
import random
from scipy.stats import truncnorm
import math


class Distribution(ABC):
    @abstractmethod
    def sample(self) -> float:
        ...

    @abstractmethod
    def mean(self) -> float:
        ...

    def __call__(self) -> float:
        return self.sample()
    
class DefiniteValue(Distribution):
    def __init__(self, value: float):
        self._value = value

    def sample(self) -> float:
        return self._value

    def mean(self) -> float:
        return self._value
    
    def __repr__(self) -> str:
        return f"DefiniteValue({self._value})"
    

class IntNormalDistribution(Distribution):
    """
    SciPy truncated normal with optional integer sampling.
    """
    def __init__(self, mean, stddev, lower=-math.inf, upper=math.inf):
        if stddev <= 0:
            raise ValueError("stddev must be positive")
        if lower > upper:
            raise ValueError("lower must be <= upper")

        self._mean = float(mean)
        self._stddev = float(stddev)
        self._lower = float(lower)
        self._upper = float(upper)

        self._rebuild()

    def _rebuild(self):
        self._a = (self._lower - self._mean) / self._stddev
        self._b = (self._upper - self._mean) / self._stddev
        self._dist = truncnorm(a=self._a, b=self._b, loc=self._mean, scale=self._stddev)


    def set_bounds(self, lower=None, upper=None):
        if lower is not None:
            self._lower = float(lower)
        if upper is not None:
            self._upper = float(upper)
        if self._lower > self._upper:
            raise ValueError("lower must be <= upper")
        self._rebuild()


    def sample(self):
        x = self._dist.rvs()

        xi = int(round(x))
        if xi < self._lower:
            xi = int(self._lower)
        if xi > self._upper:
            xi = int(self._upper)

        return xi

    def mean(self):
        return self._mean

    def __repr__(self):
        return (f"NormalDistribution(mean={self._mean}, stddev={self._stddev}, "
                f"lower={self._lower}, upper={self._upper}")
    
