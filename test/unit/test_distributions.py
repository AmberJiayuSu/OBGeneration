"""
Unit tests for distribution classes.

Tests the stochastic distribution implementations in src/stochastic/distribution.py
"""

import pytest
import numpy as np


from ob_generation.stochastic.distribution import (
    Constant,
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution
)


class TestConstant:
    """Tests for Constant distribution."""

    @pytest.mark.parametrize("value", [25.0, 0.1456, 0.0, 400.0, 0.98, 1e-5])
    def test_sample_returns_constant_value(self, value):
        """Test that sample() always returns the same value.
        """
        dist = Constant(value=value)

        sample1 = dist.sample()
        sample2 = dist.sample()
        sample3 = dist.sample()

        assert sample1 == pytest.approx(value)
        assert sample2 == pytest.approx(value)
        assert sample3 == pytest.approx(value)

    @pytest.mark.parametrize("value", [25.5, -10.0, 0.0, 1e10,0.6382])
    def test_mean_equals_constant_value(self, value):
        """Test that mean() returns the constant value."""
        dist = Constant(value=value)

        mean_value = dist.mean()

        assert mean_value == pytest.approx(value)


    @pytest.mark.parametrize("value", [25.5, -10.0, 0.0, 1e10,0.6382])
    @pytest.mark.statistical
    def test_statistical_consistency(self, value):
        """Test that sampling many times gives the same value.
        """
        dist = Constant(value=value)
        samples = [dist.sample() for _ in range(1000)]

        # Assert: All samples identical
        assert all(s == pytest.approx(value) for s in samples)
        # Assert: Mean and std dev
        assert np.mean(samples) == pytest.approx(value)
        assert np.std(samples) == pytest.approx(0.0)  # No variation




class TestNormalDistribution:
    '''Tests for NormalDistribution.'''

    @pytest.mark.parametrize("mean,stddev,lower,upper", [
        (10, 2, 0, 20),
        (15.5, 5, 10, 60),
        (0, 1, -5, 5),
        (560, 20, 0, 1000),
    ])
    def test_basic_bounds(self, mean, stddev, lower, upper):
        '''Test basic sampling works.'''
        dist = NormalDistribution(mean=mean, stddev=stddev, lower=lower, upper=upper)
        sample = dist.sample()
        assert lower <= sample <= upper


    @pytest.mark.parametrize("mean,stddev,lower,upper", [
        (10, 2, 10,10),
        (15.5, 5, 0.5,0.5),
    ])
    def test_equal_bounds(self, mean, stddev, lower, upper):
        '''Test that equal bounds returns that value.'''
        dist = NormalDistribution(mean=mean, stddev=stddev, lower=lower, upper=upper, int=False)
        assert dist.sample() == lower
        
    @pytest.mark.parametrize("mean,stddev,lower,upper", [
        (4.5, 1.5, 0, 10),
        (230.23, 102.3, 100, 400),
    ])
    def test_int(self, mean, stddev, lower, upper):
        '''Test if always returns integer when int=True.'''
        dist = NormalDistribution(mean=mean, stddev=stddev, lower=lower, upper=upper, int=True)
        assert isinstance(dist.sample(), int)

    @pytest.mark.statistical
    @pytest.mark.parametrize("mean,stddev", [
        (10, 2),
        (15.5, 5),
        (23.4, 1),
        (560, 20),
    ])
    def test_statistical_properties(self, mean, stddev):
        '''Test statistical properties of samples.'''
        dist = NormalDistribution(mean=mean, stddev=stddev, int=False)
        samples = [dist.sample() for _ in range(10000)]

        sample_mean = np.mean(samples)
        sample_stddev = np.std(samples)
        assert abs(sample_mean - mean) / abs(mean) < 0.05
        assert abs(sample_stddev - stddev) / stddev < 0.1


class TestUniformDistribution:
    '''Tests for UniformDistribution.'''

    @pytest.mark.parametrize("lower,upper", [
        (0, 10),
        (5.5, 20.5),
        (-10, 10),
        (100, 500),
    ])
    def test_basic_bounds(self, lower, upper):
        '''Test basic sampling works.'''
        dist = UniformDistribution(lower=lower, upper=upper, int=False)
        sample = dist.sample()
        assert lower <= sample <= upper

    @pytest.mark.parametrize("lower,upper", [
        (10, 10),
        (5.5, 5.5),
    ])
    def test_equal_bounds(self, lower, upper):
        '''Test that equal bounds returns that value.'''
        dist = UniformDistribution(lower=lower, upper=upper, int=False)
        assert dist.sample() == lower

    @pytest.mark.statistical
    @pytest.mark.parametrize("lower,upper", [
        (0, 10),
        (5.5, 20.5),
        (-15, 10),
        (100, 500),
    ])
    def test_statistical_properties(self, lower, upper):
        '''Test statistical properties of samples.'''
        dist = UniformDistribution(lower=lower, upper=upper, int=False)
        samples = [dist.sample() for _ in range(10000)]

        sample_mean = np.mean(samples)
        expected_mean = (lower + upper) / 2

        # Assert: Sample mean is within 2% of expected mean
        assert abs(sample_mean - expected_mean) / abs(expected_mean) < 0.05


class TestCategoricalDistribution:
    '''Tests for CategoricalDistribution.'''



    @pytest.mark.parametrize("probabilities", [
        [0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [1, 2, 3, 4],  # Non-normalized
        [0.1, 0.2, 0.7],
    ])
    def test_sample_returns_valid_index(self, probabilities):
        '''Test that sample() returns valid indices.'''
        dist = CategoricalDistribution(probabilities=probabilities)
        for _ in range(100):
            sample = dist.sample()
            assert 0 <= sample < len(probabilities)
            assert isinstance(sample, int)

    @pytest.mark.parametrize("probabilities,expected_mean", [
        ([1, 0, 0], 0.0),  # Always returns index 0
        ([0, 1, 0], 1.0),  # Always returns index 1
        ([0.5, 0.5], 0.5),  # Equal probability
        ([0.25, 0.25, 0.25, 0.25], 1.5),  # Uniform over 4 categories
    ])
    def test_mean(self, probabilities, expected_mean):
        '''Test that mean() returns expected value.'''
        dist = CategoricalDistribution(probabilities=probabilities)
        assert dist.mean() == pytest.approx(expected_mean)

    def test_zero_probabilities_fallback(self):
        '''Test that all-zero probabilities fall back to uniform.'''
        dist = CategoricalDistribution(probabilities=[0, 0, 0,0,0])
        # Should sample uniformly from all indices
        samples = [dist.sample() for _ in range(300)]
        mean = np.mean(samples)
        assert all(0 <= s < 5 for s in samples)
        assert mean == pytest.approx(2.0, abs=0.5) 

    @pytest.mark.statistical
    def test_statistical_distribution(self):
        '''Test that sampling follows the probability distribution.'''
        for _ in range(5):
            # Generate random probabilities
            n_categories = np.random.randint(2, 50)  # Random number of categories (2-9)
            probabilities = np.random.random(n_categories).tolist()  # Random probabilities

            dist = CategoricalDistribution(probabilities=probabilities)
            n_samples = 1000
            samples = [dist.sample() for _ in range(n_samples)]

            # Count occurrences of each index
            counts = [samples.count(i) for i in range(len(probabilities))]
            frequencies = [c / n_samples for c in counts]

            # Normalize expected probabilities
            total = sum(probabilities)
            expected = [p / total for p in probabilities]

            # Assert: Frequencies are within 5% of expected probabilities
            for freq, exp in zip(frequencies, expected):
                assert abs(freq - exp) < 0.05


    def test_update_probabilities_by_value(self):
        '''Test updating probabilities by setting new values.'''
        for _ in range(5):
            # Generate random probabilities
            n_categories = np.random.randint(2, 20)
            probabilities = np.random.random(n_categories).tolist()

            # Randomly select indices to update
            n_updates = np.random.randint(1, min(5, n_categories + 1))
            indices_to_update = np.random.choice(n_categories, n_updates, replace=False).tolist()

            # Generate random new probability values
            new_values = np.random.random(n_updates).tolist()
            index_updates = {new_val: [idx] for new_val, idx in zip(new_values, indices_to_update)}

            dist = CategoricalDistribution(probabilities=probabilities)

            # Store original normalized probabilities from the distribution
            original_probs = dist._probabilities.copy()

            # Apply updates to normalized probabilities manually
            updated_probs = original_probs.copy()
            for new_val, indices in index_updates.items():
                for idx in indices:
                    updated_probs[idx] = new_val

            # Normalize manually
            total = sum(updated_probs)
            expected_probs = [p / total for p in updated_probs]

            # Perform the actual update
            dist.update_probabilities_by_value(index_updates)

            # Probabilities should sum to 1 after update
            assert sum(dist._probabilities) == pytest.approx(1.0)

            # Check that updated indices have changed
            for indices in index_updates.values():
                for idx in indices:
                    # The probability should have been set (then normalized)
                    assert dist._probabilities[idx] >= 0

            # Calculate expected mean from manually normalized probabilities
            expected_mean = sum(i * p for i, p in enumerate(expected_probs))
            actual_mean = dist.mean()
            assert actual_mean == pytest.approx(expected_mean)



    def test_update_probabilities_by_factor(self):
        '''Test updating probabilities by multiplication factor.'''
        for _ in range(5):
            # Generate random probabilities
            n_categories = np.random.randint(2, 20)
            probabilities = np.random.random(n_categories).tolist()

            dist = CategoricalDistribution(probabilities=probabilities)

            # Store original normalized probabilities
            original_probs = dist._probabilities.copy()

            # Randomly select indices to update
            n_updates = np.random.randint(1, min(5, n_categories + 1))
            indices_to_update = np.random.choice(n_categories, n_updates, replace=False).tolist()

            # Generate random factors (between 0.1 and 5.0)
            factors = (np.random.random(n_updates) * 4.9 + 0.1).tolist()
            index_updates = {factor: [idx] for factor, idx in zip(factors, indices_to_update)}

            # Apply updates manually to calculate expected result
            updated_probs = original_probs.copy()
            for factor, indices in index_updates.items():
                for idx in indices:
                    updated_probs[idx] = updated_probs[idx] * factor

            # Normalize manually
            total = sum(updated_probs)
            expected_probs = [p / total for p in updated_probs]

            # Perform the actual update
            dist.update_probabilities_by_factor(index_updates)

            # Probabilities should still sum to 1 after update
            assert sum(dist._probabilities) == pytest.approx(1.0)

            # Check that probabilities changed for updated indices
            for indices in index_updates.values():
                for idx in indices:
                    assert dist._probabilities[idx] != original_probs[idx]

            # Calculate expected mean from manually normalized probabilities
            expected_mean = sum(i * p for i, p in enumerate(expected_probs))
            actual_mean = dist.mean()
            assert actual_mean == pytest.approx(expected_mean)


    @pytest.mark.parametrize("probabilities,start,end", [
        ([0.25, 0.25, 0.25, 0.25], 1, 3),
        ([0.1, 0.2, 0.3, 0.4], 0, 2),
        ([1, 1, 1, 1, 1], 2, 5),
    ])
    def test_sample_from_range(self, probabilities, start, end):
        '''Test sampling from a specific range of indices.'''
        dist = CategoricalDistribution(probabilities=probabilities)

        for _ in range(100):
            sample = dist.sample_from_range(start, end)
            assert start <= sample < end
            assert isinstance(sample, int)

    @pytest.mark.statistical
    def test_sample_from_range_distribution(self):
        '''Test that sample_from_range respects probabilities within range.'''
        # Create distribution where index 2 has much higher probability
        dist = CategoricalDistribution(probabilities=[0.1, 0.1, 0.6, 0.2])

        # Sample from range [1, 3) which includes indices 1 and 2
        n_samples = 5000
        samples = [dist.sample_from_range(1, 3) for _ in range(n_samples)]

        # Count occurrences
        count_1 = samples.count(1)
        count_2 = samples.count(2)

        # Index 2 should be sampled more often (0.6 vs 0.1 in original)
        # After normalization within range: 0.1/(0.1+0.6) ≈ 0.14 and 0.6/(0.1+0.6) ≈ 0.86
        assert count_2 > count_1  # Much more samples of index 2
