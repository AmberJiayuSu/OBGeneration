"""
Unit tests for DistributionConfig.

Tests the configuration builder in src/stochastic/distribution_config.py
"""

import pytest
from stochastic.distribution_config import DistributionConfig
from stochastic.distribution import (
    Constant,
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution
)


class TestConfig:
    """Tests for constant distribution configuration."""

    def test_build_constant(self):
        """Test building a constant distribution from config."""
        config = DistributionConfig(
            dist_type="constant",
            params={"value": 42.5}
        )
        dist = config.build()

        assert isinstance(dist, Constant)
        assert dist.sample() == pytest.approx(42.5)
        assert dist.mean() == pytest.approx(42.5)


    def test_build_uniform_with_custom_bounds(self):
        """Test building uniform distribution with custom bounds."""
        config = DistributionConfig(
            dist_type="uniform",
            params={"min": 10, "max": 20, "int": False}
        )
        dist = config.build()

        assert isinstance(dist, UniformDistribution)
        for _ in range(20):
            sample = dist.sample()
            assert 10 <= sample <= 20


    def test_build_normal_with_parameters(self):
        """Test building normal distribution with custom parameters."""
        config = DistributionConfig(
            dist_type="normal",
            params={"mean": 100, "std": 15, "lower": 50, "upper": 150, "int": False}
        )
        dist = config.build()

        assert isinstance(dist, NormalDistribution)
        assert dist.mean() == pytest.approx(100.0)

        # Check bounds
        for _ in range(50):
            sample = dist.sample()
            assert 50 <= sample <= 150


    def test_build_categorical_distribution(self):
        """Test building a categorical distribution from config."""
        config = DistributionConfig(
            dist_type="categorical",
            params={"weights": [0.2, 0.5, 0.3]}
        )
        dist = config.build()

        assert isinstance(dist, CategoricalDistribution)

        # Check that sampling respects weights (basic check)
        counts = [0, 0, 0]
        num_samples = 1000
        for _ in range(num_samples):
            sample = dist.sample()
            counts[sample] += 1

        # The counts should roughly reflect the weights
        assert abs(counts[0] / num_samples - 0.2) < 0.1
        assert abs(counts[1] / num_samples - 0.5) < 0.1
        assert abs(counts[2] / num_samples - 0.3) < 0.1
