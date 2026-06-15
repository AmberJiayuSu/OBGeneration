from abc import ABC, abstractmethod
from math import erf, sqrt
import numpy as np
from scipy.special import erfinv


class Distribution(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator | None = None) -> float:
        ...

    @abstractmethod
    def mean(self) -> float:
        ...

    def __call__(self) -> float:
        return self.sample()


class Constant(Distribution):
    def __init__(self, value: float):
        self._value = value

    def sample(self, rng: np.random.Generator | None = None) -> float:
        return self._value

    def mean(self) -> float:
        return self._value

    def __repr__(self) -> str:
        return f"Constant({self._value})"


class NormalDistribution(Distribution):
    """Truncated normal using inverse CDF sampling. All randomness via numpy Generator."""

    @staticmethod
    def _cdf(x: float) -> float:
        if x == np.inf:
            return 1.0
        if x == -np.inf:
            return 0.0
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def _icdf(p: float) -> float:
        p = max(1e-12, min(1.0 - 1e-12, p))
        return sqrt(2.0) * erfinv(2.0 * p - 1.0)

    def __init__(self, mean, stddev, lower=-np.inf, upper=np.inf, as_int: bool = True):
        if stddev <= 0:
            raise ValueError("stddev must be positive")
        if lower > upper:
            raise ValueError(f"lower: {lower} must be <= upper: {upper}")
        self._int = as_int
        self._mean = float(mean)
        self._stddev = float(stddev)
        self._lower = float(lower)
        self._upper = float(upper)
        # Precompute CDF bounds for sampling
        self._p_low = self._cdf((self._lower - self._mean) / self._stddev)
        self._p_high = self._cdf((self._upper - self._mean) / self._stddev)

    def _recompute_bounds(self):
        self._p_low = self._cdf((self._lower - self._mean) / self._stddev)
        self._p_high = self._cdf((self._upper - self._mean) / self._stddev)

    def set_bounds(self, lower=None, upper=None):
        if lower is not None:
            self._lower = float(lower)
        if upper is not None:
            self._upper = float(upper)
        if self._lower > self._upper:
            raise ValueError("lower must be <= upper")
        self._recompute_bounds()

    def with_bounds(self, lower=None, upper=None) -> "NormalDistribution":
        """Returns a new NormalDistribution with updated bounds without mutating the original."""
        return NormalDistribution(
            mean=self._mean,
            stddev=self._stddev,
            lower=float(lower) if lower is not None else self._lower,
            upper=float(upper) if upper is not None else self._upper,
            as_int=self._int,
        )

    def update_mean(self, mean: float):
        self._mean = float(mean)
        self._recompute_bounds()

    def sample(self, rng: np.random.Generator | None = None) -> float:
        if self._lower == self._upper:
            return int(self._lower) if self._int else self._lower
        _rng = rng if rng is not None else np.random.default_rng()
        u = float(_rng.uniform(self._p_low, self._p_high))
        x = self._mean + self._stddev * self._icdf(u)
        x = max(self._lower, min(self._upper, x))
        return int(round(x)) if self._int else x

    def mean(self) -> float:
        return self._mean

    def __repr__(self) -> str:
        return (f"NormalDistribution(mean={self._mean}, stddev={self._stddev}, "
                f"lower={self._lower}, upper={self._upper})")


class UniformDistribution(Distribution):
    def __init__(self, lower: float, upper: float, as_int: bool = True):
        if lower > upper:
            raise ValueError("lower must be <= upper")
        self._lower = float(lower)
        self._upper = float(upper)
        self._int = as_int

    def sample(self, rng: np.random.Generator | None = None) -> float:
        _rng = rng if rng is not None else np.random.default_rng()
        x = _rng.uniform(self._lower, self._upper)
        return int(round(x)) if self._int else x

    def mean(self) -> float:
        return (self._lower + self._upper) / 2

    def __repr__(self) -> str:
        return f"UniformDistribution(lower={self._lower}, upper={self._upper}, int={self._int})"


class BinomialDistribution(Distribution):
    """Binomial distribution: number of successes in n trials with probability p."""

    def __init__(self, n: int, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if n < 0:
            raise ValueError("n must be non-negative")
        self._n = n
        self._p = p

    def sample(self, rng: np.random.Generator | None = None) -> int:
        _rng = rng if rng is not None else np.random.default_rng()
        return int(_rng.binomial(self._n, self._p))

    def mean(self) -> float:
        return self._n * self._p

    def __repr__(self) -> str:
        return f"BinomialDistribution(n={self._n}, p={self._p})"


class CategoricalDistribution(Distribution):
    """Samples indices based on probability weights."""

    def __init__(self, probabilities: list[float]):
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-6 and total > 0:
            probabilities = [p / total for p in probabilities]
        self._probabilities = probabilities
        self._total = sum(probabilities)
        self._cumsum = self._build_cumsum()

    def _build_cumsum(self) -> list[float]:
        cumsum, cumulative = [], 0.0
        for p in self._probabilities:
            cumulative += p / self._total if self._total > 0 else 1.0 / len(self._probabilities)
            cumsum.append(cumulative)
        return cumsum

    def _renormalize(self):
        self._total = sum(self._probabilities)
        if self._total > 0:
            self._probabilities = [p / self._total for p in self._probabilities]
        else:
            n = len(self._probabilities)
            self._probabilities = [1.0 / n for _ in self._probabilities]
        self._total = sum(self._probabilities)
        self._cumsum = self._build_cumsum()

    def sample(self, rng: np.random.Generator | None = None) -> int:
        _rng = rng if rng is not None else np.random.default_rng()
        if self._total == 0:
            return int(_rng.integers(0, len(self._probabilities)))
        r = float(_rng.random())
        for i, cum_prob in enumerate(self._cumsum):
            if r <= cum_prob:
                return i
        return len(self._probabilities) - 1

    def sample_from_range(self, start: int, end: int, rng: np.random.Generator | None = None) -> int:
        _rng = rng if rng is not None else np.random.default_rng()
        if self._total == 0 or start >= end or start < 0 or end > len(self._probabilities):
            return int(_rng.integers(start, end))
        range_total = sum(self._probabilities[start:end])
        if range_total == 0:
            return int(_rng.integers(start, end))
        range_cumsum, cumulative = [], 0.0
        for i in range(start, end):
            cumulative += self._probabilities[i] / range_total
            range_cumsum.append(cumulative)
        r = float(_rng.random())
        for i, cum_prob in enumerate(range_cumsum):
            if r <= cum_prob:
                return start + i
        return end - 1

    def update_probabilities_by_value(self, index_to_update: dict[float, list[int]]):
        for new_prob, indices in index_to_update.items():
            for index in indices:
                if 0 <= index < len(self._probabilities):
                    self._probabilities[index] = new_prob
        self._renormalize()

    def update_probabilities_by_factor(self, index_to_update: dict[float, list[int]]):
        for change_factor, indices in index_to_update.items():
            for index in indices:
                if 0 <= index < len(self._probabilities):
                    self._probabilities[index] *= change_factor
        self._renormalize()

    def mean(self) -> float:
        if self._total == 0:
            return len(self._probabilities) / 2
        return sum(i * p for i, p in enumerate(self._probabilities)) / self._total

    def __repr__(self) -> str:
        return f"CategoricalDistribution(n_categories={len(self._probabilities)})"
