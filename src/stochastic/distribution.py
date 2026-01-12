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
    
class Constant(Distribution):
    def __init__(self, value: float):
        self._value = value

    def sample(self) -> float:
        return self._value

    def mean(self) -> float:
        return self._value
    
    def __repr__(self) -> str:
        return f"Constant({self._value})"
    

class NormalDistribution(Distribution):
    """
    SciPy truncated normal with optional integer sampling.
    """
    def __init__(self, mean, stddev, lower=-math.inf, upper=math.inf,int: bool = True):
        self._int = int
        if stddev <= 0:
            raise ValueError("stddev must be positive")
        if lower > upper:
            raise ValueError(fr"lower: {lower} must be <= upper: {upper}")

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

    def with_bounds(self, lower=None, upper=None):
        """
        Returns a new NormalDistribution with updated bounds without mutating the original.
        """
        new_lower = float(lower) if lower is not None else self._lower
        new_upper = float(upper) if upper is not None else self._upper
        return NormalDistribution(
            mean=self._mean,
            stddev=self._stddev,
            lower=new_lower,
            upper=new_upper,
            int=self._int
        )

    def update_mean(self, mean: float):
        self._mean = float(mean)
        self._rebuild()


    def sample(self):
        if self._lower == self._upper:
            return int(self._lower) if self._int else self._lower
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


class CategoricalDistribution(Distribution):
    """Samples indices based on probability weights (like a discrete probability distribution)."""
    def __init__(self, probabilities: list[float]):
        """
        Args:
            probabilities: List of probabilities (will be normalized if they don't sum to 1)
        """
        total = sum(probabilities)
        if math.fabs(total - 1.0) > 1e-6 and total > 0:
            probabilities = [p / total for p in probabilities]
        self._probabilities = probabilities
        self._total = sum(probabilities)

        # Precompute cumulative sum for efficient sampling
        self._cumsum = []
        cumulative = 0.0
        for p in probabilities:
            cumulative += p / self._total if self._total > 0 else 1.0 / len(probabilities)
            self._cumsum.append(cumulative)

    def sample(self) -> int:
        """Sample an index based on the probability distribution.
        Returns:
            Index (0 to len(probabilities)-1)
        """
        if self._total == 0:
            return random.randint(0, len(self._probabilities) - 1)

        r = random.random()
        # Binary search through cumsum
        for i, cum_prob in enumerate(self._cumsum):
            if r <= cum_prob:
                return i
        return len(self._probabilities) - 1
    
    def update_probabilities_by_value(self, index_to_update: dict[float, list[int]]):
        for new_prob, indices in index_to_update.items():
            for index in indices:
                if 0 <= index < len(self._probabilities):
                    self._probabilities[index] = new_prob
        self._total = sum(self._probabilities)
        #renormalize
        if self._total > 0:
            self._probabilities = [p / self._total for p in self._probabilities]
        else:
            #fallback to uniform if all probabilities are zero
            n = len(self._probabilities)
            self._probabilities = [1.0 / n for _ in self._probabilities]
        self._total = sum(self._probabilities)
        # Recompute cumulative sum
        self._cumsum = []
        cumulative = 0.0
        for p in self._probabilities:
            cumulative += p
            self._cumsum.append(cumulative)

    def update_probabilities_by_factor(self, index_to_update: dict[float, list[int]]):
        for change_factor, indices in index_to_update.items():
            for index in indices:
                if 0 <= index < len(self._probabilities):
                    self._probabilities[index] = self._probabilities[index] * change_factor
        self._total = sum(self._probabilities)
        #renormalize
        if self._total > 0:
            self._probabilities = [p / self._total for p in self._probabilities]
        else:
            #fallback to uniform if all probabilities are zero
            n = len(self._probabilities)
            self._probabilities = [1.0 / n for _ in self._probabilities]
        self._total = sum(self._probabilities)
        # Recompute cumulative sum
        self._cumsum = []
        cumulative = 0.0
        for p in self._probabilities:
            cumulative += p
            self._cumsum.append(cumulative)

    def sample_from_range(self, start: int, end: int) -> int:
        """Sample an index within a specific range [start, end).
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
        Returns:
            Index (start to end-1)
        """
        if self._total == 0 or start >= end or start < 0 or end > len(self._probabilities):
            return random.randint(start, end - 1)

        # Compute cumulative sum for the specified range
        range_total = sum(self._probabilities[start:end])
        if range_total == 0:
            return random.randint(start, end - 1)

        range_cumsum = []
        cumulative = 0.0
        for i in range(start, end):
            cumulative += self._probabilities[i] / range_total
            range_cumsum.append(cumulative)

        r = random.random()
        for i, cum_prob in enumerate(range_cumsum):
            if r <= cum_prob:
                return start + i
        return end - 1


    def mean(self) -> float:
        """Expected value of the index."""
        if self._total == 0:
            return len(self._probabilities) / 2
        return sum(i * p for i, p in enumerate(self._probabilities)) / self._total

    def __repr__(self) -> str:
        return f"CategoricalDistribution(n_categories={len(self._probabilities)})"

