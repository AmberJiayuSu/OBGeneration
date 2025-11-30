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
    

class NormalDistribution(Distribution):
    """
    SciPy truncated normal with optional integer sampling.
    """
    def __init__(self, mean, stddev, lower=-math.inf, upper=math.inf,int: bool = True):
        self._int = int
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

        xi = int(round(x)) if self._int else x
        if xi < self._lower:
            xi = int(self._lower) if self._int else self._lower
        if xi > self._upper:
            xi = int(self._upper) if self._int else self._upper
        return xi

    def mean(self):
        return self._mean

    def __repr__(self):
        return (f"NormalDistribution(mean={self._mean}, stddev={self._stddev}, "
                f"lower={self._lower}, upper={self._upper}")
    

class UniformDistribution(Distribution):
    def __init__(self, lower: float, upper: float,int: bool = True):
        if lower > upper:
            raise ValueError("lower must be <= upper")
        self._lower = float(lower)
        self._upper = float(upper)
        self._int = int

    def sample(self) -> float:
        x = random.uniform(self._lower, self._upper)
        return int(round(x)) if self._int else x

    def mean(self) -> float:
        return (self._lower + self._upper) / 2

    def __repr__(self) -> str:
        return (f"UniformDistribution(lower={self._lower}, "
                f"upper={self._upper}, int={self._int})")


class WeightedValueDistribution(Distribution):
    def __init__(self, values_with_weights: dict[float, list[float]]):
        self._values = []
        self._weights = []
        for weight, vals in values_with_weights.items():
            for v in vals:
                self._values.append(v)
                self._weights.append(weight)
    
    def update_weights(self, weight_to_update: dict[float, list[float]]):
        for weight, vals in weight_to_update.items():
            for v in vals:
                if v in self._values:
                    idx = self._values.index(v)
                    self._weights[idx] = weight
    
    def update_weights_by_factor(self, weight_to_update: dict[float, list[float]]):
        for change_factor, vals in weight_to_update.items():
            for v in vals:
                if v in self._values:
                    idx = self._values.index(v)
                    self._weights[idx] = self._weights[idx] * change_factor
                    
    def sample(self) -> float:
        if sum(self._weights) == 0:
            return None
        return random.choices(self._values, weights=self._weights, k=1)[0]
    
    def mean(self) -> float:
        total_weight = sum(self._weights)
        weighted_sum = sum(v * w for v, w in zip(self._values, self._weights))
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def __repr__(self) -> str:
        return f"AssignedValueDistribution(values={self._values}, weights={self._weights})"