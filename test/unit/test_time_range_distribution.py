import numpy as np
import pytest

from obgeneration.generator.occupancy_generator import TimeRangeDistribution
from obgeneration.model.occupancy import TimeRange


SIM_RES = 15


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def away_range_typical() -> TimeRange:
    return TimeRange(start_hour=9, end_hour=17)


@pytest.fixture
def away_range_wraps_midnight() -> TimeRange:
    return TimeRange(start_hour=22, end_hour=6)


class TestTimeRange:
    @pytest.mark.parametrize("value, resolution, expected", [
        (8.1, 0.25, 8.0),
        (8.2, 0.25, 8.25),
        (8.4, 0.25, 8.5),
        (0.1, 0.5, 0.0),
        (23.9, 1.0, 23.0),
        (12.33, 0.1, 12.3),
        (5.67, 0.2, 5.6),
    ])
    def test_snap_to_resolution(self, value, resolution, expected):
        assert TimeRangeDistribution._snap(value, resolution) == pytest.approx(expected)

    @pytest.mark.parametrize("start, end", [
        (22.0, 2.0),
        (23.5, 0.5),
    ])
    def test_wraps_midnight_handling(self, start, end, rng):
        wrapping_range = TimeRange(start_hour=start, end_hour=end)
        assert wrapping_range.wraps_midnight is True

        dist = TimeRangeDistribution(wrapping_range, start_variance=0.5, end_variance=0.5, resolution_mins=15)
        assert dist.wraps_midnight is True
        sampled = dist.sample(rng)
        assert isinstance(sampled, TimeRange)

    @pytest.mark.statistical
    @pytest.mark.parametrize("start, end", [
        (9.0, 17.0),
        (8.5, 16.5),
        (10.0, 15.0),
    ])
    def test_sample_returns_valid_time_range(self, start, end, rng):
        base_range = TimeRange(start_hour=start, end_hour=end)
        dist = TimeRangeDistribution(base_range, start_variance=1.0, end_variance=1.0, resolution_mins=15)

        starts = []
        ends = []
        for _ in range(100):
            sampled = dist.sample(rng)
            starts.append(sampled.start_hour)
            ends.append(sampled.end_hour)
            assert isinstance(sampled, TimeRange)
            assert 0 <= sampled.start_hour < 24
            assert 0 <= sampled.end_hour < 24
            if not sampled.wraps_midnight:
                assert sampled.end_hour > sampled.start_hour

        start_mean = np.mean(starts)
        end_mean = np.mean(ends)
        assert abs(start_mean - start) / abs(start) < 0.05
        assert abs(end_mean - end) / abs(end) < 0.05

        start_std = np.std(starts)
        end_std = np.std(ends)
        assert abs(start_std - 1.0) < 0.3
        assert abs(end_std - 1.0) < 0.3

    @pytest.mark.parametrize("start, end, res, var", [
        (22.0, 4.0, 15, 1.0),
        (9.15, 17.45, 30, 3.0),
        (8.5, 16.5, 60, 2.0),
    ])
    def test_sample_many_times_stays_valid(self, start, end, res, var, rng):
        base_range = TimeRange(start_hour=start, end_hour=end)
        dist = TimeRangeDistribution(base_range, start_variance=var, end_variance=var, resolution_mins=res)
        start_vals = []
        end_vals = []
        for _ in range(100):
            sampled = dist.sample(rng)
            start_vals.append(sampled.start_hour)
            end_vals.append(sampled.end_hour)
            assert isinstance(sampled, TimeRange)
            assert 0 <= sampled.start_hour < 24
            assert 0 <= sampled.end_hour < 24
            if not sampled.wraps_midnight:
                assert sampled.end_hour > sampled.start_hour
        mean_tol = max(0.5, var * 0.3)
        assert np.mean(start_vals) == pytest.approx(start, abs=mean_tol)
        assert np.mean(end_vals) == pytest.approx(end, abs=mean_tol)
        assert np.std(start_vals) == pytest.approx(var, abs=0.5)
        assert np.std(end_vals) == pytest.approx(var, abs=0.5)


class TestTimeRangeDistributionSampleWithAwayBounds:
    @pytest.mark.parametrize("start, end", [
        (10.0, 18.0),
        (7.0, 14.0),
    ])
    def test_returns_none_when_mean_in_away_range_day(self, start, end, away_range_typical, rng):
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(rng, away_range_typical)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (23.0, 8.0),
        (21.0, 4.0),
    ])
    def test_returns_none_when_mean_in_away_range_night(self, start, end, away_range_wraps_midnight, rng):
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(rng, away_range_wraps_midnight)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (11.0, 14.0),
        (7.0, 20.0),
    ])
    def test_returns_none_when_fully_contains_day(self, start, end, away_range_typical, rng):
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(rng, away_range_typical)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (19.0, 8.0),
        (1.0, 5.0),
    ])
    def test_returns_none_when_fully_contains_night(self, start, end, away_range_wraps_midnight, rng):
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(rng, away_range_wraps_midnight)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (22.0, 6.0),
        (1.0, 8.0),
    ])
    def test_sampled_range_outside_away_window_day(self, start, end, away_range_typical, rng):
        for _ in range(50):
            dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
            sampled = dist.sample_with_away_bounds(rng, away_range_typical)
            assert sampled is not None
            assert not sampled.overlaps(away_range_typical)

    @pytest.mark.parametrize("start, end", [
        (8.0, 15.0),
        (12.0, 20.0),
    ])
    def test_sampled_range_outside_away_window_night(self, start, end, away_range_wraps_midnight, rng):
        for _ in range(50):
            dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
            sampled = dist.sample_with_away_bounds(rng, away_range_wraps_midnight)
            assert sampled is not None
            assert not sampled.overlaps(away_range_wraps_midnight)
