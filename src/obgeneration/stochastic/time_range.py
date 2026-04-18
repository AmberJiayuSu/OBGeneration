"""Stochastic sampling of time ranges within a day."""

from typing import Optional

import numpy as np

from obgeneration.model.occupancy import TimeRange
from obgeneration.stochastic.distribution import NormalDistribution


class TimeRangeDistribution:
    """Samples a TimeRange (start/end hours) from two truncated normal distributions."""

    def __init__(
        self,
        time_range: TimeRange,
        start_variance: float,
        end_variance: float,
        resolution_mins: int = 15,
    ):
        self.resolution_mins = resolution_mins
        self.resolution_hours = resolution_mins / 60
        self.time_range = time_range
        self.wraps_midnight = time_range.wraps_midnight
        upper_bound = 24.0 - self.resolution_hours
        self.start_dist = NormalDistribution(
            mean=time_range.start_hour,
            stddev=start_variance,
            lower=0.0,
            upper=upper_bound,
            as_int=False,
        )
        self.end_dist = NormalDistribution(
            mean=time_range.end_hour,
            stddev=end_variance,
            lower=0.0,
            upper=upper_bound,
            as_int=False,
        )

    @staticmethod
    def _snap(time: float, resolution_hours: float) -> float:
        snapped = round(time / resolution_hours) * resolution_hours
        return min(snapped, 24.0 - resolution_hours)

    def overlaps(self, other: TimeRange) -> bool:
        if other.contains_hour(self.start_dist.mean()) or other.contains_hour(self.end_dist.mean()):
            return True
        if self.time_range.contains_hour(other.start_hour) and self.time_range.contains_hour(other.end_hour):
            return True
        return False

    def sample(self, rng: np.random.Generator) -> TimeRange:
        """Sample start/end hours, retrying until end > start (unless wraps midnight)."""
        MAX_ATTEMPTS = 1000
        INNER_ATTEMPTS = 20
        total_attempts = 0
        while total_attempts < MAX_ATTEMPTS:
            start_time = self._snap(self.start_dist.sample(rng), self.resolution_hours)
            for _ in range(INNER_ATTEMPTS):
                end_time = self._snap(self.end_dist.sample(rng), self.resolution_hours)
                if self.wraps_midnight or end_time > start_time:
                    return TimeRange(start_hour=start_time, end_hour=end_time)
            total_attempts += INNER_ATTEMPTS
        return TimeRange(start_hour=start_time, end_hour=end_time)

    def sample_with_away_bounds(
        self, rng: np.random.Generator, away_time: Optional[TimeRange]
    ) -> Optional[TimeRange]:
        """Sample a time range that does not overlap with away_time."""
        if away_time is None:
            return self.sample(rng)

        if self.overlaps(away_time):
            return None

        if not away_time.wraps_midnight:
            if self.wraps_midnight:
                start_dist = self.start_dist.with_bounds(
                    lower=away_time.end_hour + self.resolution_hours,
                    upper=24.0 - self.resolution_hours,
                )
                end_dist = self.end_dist.with_bounds(
                    lower=0.0,
                    upper=away_time.start_hour - self.resolution_hours,
                )
            else:
                start_dist = self.start_dist.with_bounds(
                    lower=0.0,
                    upper=away_time.start_hour - self.resolution_hours,
                )
                end_dist = self.end_dist.with_bounds(
                    lower=0.0,
                    upper=away_time.start_hour - self.resolution_hours,
                )
        else:
            start_dist = self.start_dist.with_bounds(
                lower=away_time.end_hour + self.resolution_hours,
                upper=away_time.start_hour - self.resolution_hours,
            )
            end_dist = self.end_dist.with_bounds(
                lower=away_time.end_hour + self.resolution_hours,
                upper=away_time.start_hour - self.resolution_hours,
            )

        MAX_ATTEMPTS = 1000
        INNER_ATTEMPTS = 20
        total_attempts = 0
        while total_attempts < MAX_ATTEMPTS:
            start_time = self._snap(start_dist.sample(rng), self.resolution_hours)
            for _ in range(INNER_ATTEMPTS):
                end_time = self._snap(end_dist.sample(rng), self.resolution_hours)
                if self.wraps_midnight or end_time > start_time:
                    return TimeRange(start_hour=start_time, end_hour=end_time)
            total_attempts += INNER_ATTEMPTS
        return TimeRange(start_hour=start_time, end_hour=end_time)
