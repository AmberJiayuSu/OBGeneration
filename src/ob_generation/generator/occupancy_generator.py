from ob_generation.stochastic.distribution import Distribution,NormalDistribution
from ob_generation.model.occupancy import Occupancy, TimeRange, WeekendOccupancyPattern, WeekdayOccupancyPattern
from ob_generation.stochastic.distribution_config import DistributionConfig
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import json
from pathlib import Path
from typing import Optional, NamedTuple, Tuple
from ob_generation.generator.ob_utils import ScheduleUtils
from importlib.resources import files
import pandas as pd


class TimeRangeDistribution:
    """Distribution for time ranges within a day."""
    def __init__(self, time_range: TimeRange, start_variance:float, end_variance:float, resolution_mins: int = 15):
        self.resolution_mins = resolution_mins
        self.resolution_hours = resolution_mins / 60
        self.time_range = time_range
        upper_bound = 24.0 - self.resolution_hours
        self.start_dist = NormalDistribution(
            mean= time_range.start_hour ,
            stddev=start_variance,
            lower=0.0,
            upper=upper_bound,
            as_int=False
        )
        self.end_dist = NormalDistribution(
            mean= time_range.end_hour ,
            stddev=end_variance,
            lower=0.0,
            upper=upper_bound,
            as_int=False
        )
        self.wraps_midnight = time_range.wraps_midnight

    @staticmethod
    def _snap_to_resolution( time: float, resolution_hours:float) -> float:
        """Snap a time value to the nearest resolution point."""
        snapped = round(time / resolution_hours) * resolution_hours
        return min(snapped, 24.0 - resolution_hours)


    def sample(self) -> TimeRange:
        """
        Samples a single occupant's leave and return times ensuring logical consistency.

        Args:
            start_dist: Distribution for leave time
            end_dist: Distribution for return time
            wraps_midnight: If True, allows return time < leave time (crosses midnight)

        Returns:
            Tuple of (start_time, end_time)
        """
        MAX_ATTEMPTS = 1000
        INNER_ATTEMPTS = 20

        total_attempts = 0
        while total_attempts < MAX_ATTEMPTS:
            start_time = TimeRangeDistribution._snap_to_resolution(self.start_dist.sample(), self.resolution_hours)
            for _ in range(INNER_ATTEMPTS):
                end_time = TimeRangeDistribution._snap_to_resolution(self.end_dist.sample(), self.resolution_hours)
                if self.wraps_midnight or end_time > start_time:
                    return TimeRange(start_hour=start_time, end_hour=end_time)
            total_attempts += INNER_ATTEMPTS

        # Fallback: if bounds make it impossible to satisfy end > start,
        # allow end >= start to avoid infinite loop
        return TimeRange(start_hour=start_time, end_hour=end_time)
    
    def overlaps(self, other: TimeRange) -> bool:
        """Check if this distribution's time range overlaps with another TimeRange."""
        if other.contains_hour(self.start_dist.mean()) or other.contains_hour(self.end_dist.mean()):
            return True
        if self.time_range.contains_hour(other.start_hour) and self.time_range.contains_hour(other.end_hour):
            return True
        return False
 

    def sample_with_away_bounds( self,away_time: Optional[TimeRange]) -> Optional[TimeRange]:
        """
        Samples a bound that is outside the away_time if provided, ensuring logical consistency.

        Args:
            away_time: TimeRange during which the occupant is away. If None, no bounds are applied.
            resolution_hours: The time resolution in hours to which the sampled times should be snapped.
        """
        if away_time is None:
            return self.sample()
        else:
            if self.overlaps(away_time):
                return None  # No valid time range can be sampled outside the away_time
            
            if not away_time.wraps_midnight:
                if self.wraps_midnight:
                    updated_start_dist = self.start_dist.with_bounds(lower=away_time.end_hour + self.resolution_hours, upper=24.0 - self.resolution_hours)
                    updated_end_dist = self.end_dist.with_bounds(lower=0.0, upper=away_time.start_hour - self.resolution_hours)
                else:
                    updated_start_dist = self.start_dist.with_bounds(lower=0.0, upper=away_time.start_hour - self.resolution_hours)
                    updated_end_dist = self.end_dist.with_bounds(lower=0.0, upper=away_time.start_hour - self.resolution_hours)
            else:
                # Away time wraps midnight, so at this branch the self time range cannot wrap midnight
                updated_start_dist = self.start_dist.with_bounds(lower=away_time.end_hour + self.resolution_hours, upper=away_time.start_hour - self.resolution_hours)
                updated_end_dist = self.end_dist.with_bounds(lower=away_time.end_hour + self.resolution_hours, upper=away_time.start_hour - self.resolution_hours)

        MAX_ATTEMPTS = 1000
        INNER_ATTEMPTS = 20

        total_attempts = 0
        while total_attempts < MAX_ATTEMPTS:
            start_time = TimeRangeDistribution._snap_to_resolution(updated_start_dist.sample(), resolution_hours=self.resolution_hours)
            for _ in range(INNER_ATTEMPTS):
                end_time = TimeRangeDistribution._snap_to_resolution(updated_end_dist.sample(), resolution_hours=self.resolution_hours)
                if self.wraps_midnight or end_time > start_time:
                    return TimeRange(start_hour=start_time, end_hour=end_time)
            total_attempts += INNER_ATTEMPTS

        return TimeRange(start_hour=start_time, end_hour=end_time)



class OccupancyState(Enum):
    AWAY = 0
    HOME = 1
    SLEEP = 2


class ClusterAssumptions:
    """
    Stores Markov-chain parameters for occupancy state simulation.

    States are defined by OccupancyState:
        AWAY  = 0
        HOME  = 1
        SLEEP = 2

    The day is divided into (1440 / resolution_min) bins, starting at start_min.
    With the defaults (start_min=240, resolution_min=15):
        bin 0  → 04:00–04:15
        bin 1  → 04:15–04:30
        ...
        bin 95 → 03:45–04:00 (next day)

    Attributes
    ----------
    num_clusters : int
        Number of household behaviour clusters.

    start_min : int
        The minute-of-day at which bin 0 begins (e.g. 240 = 04:00).
        Bins wrap around midnight, so the sequence covers a full 24-hour cycle
        starting from this offset.

    resolution_min : int
        Duration of each bin in minutes (e.g. 15). Must evenly divide 1440.
        Determines the total number of bins: num_bins = 1440 // resolution_min.

    weekday_initial_probs : list[list[float]], shape [num_clusters, 3]
        weekday_initial_probs[cluster][state] = P(initial state | cluster) on a weekday.
        State index follows OccupancyState: AWAY=0, HOME=1, SLEEP=2.

    weekend_initial_probs : list[list[float]], shape [num_clusters, 3]
        Same as weekday_initial_probs but for weekends.

    weekday_transition_probs : dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]}
        weekday_transition_probs[cluster][bin][from_state][to_state]
            = P(next state = to_state | current state = from_state, time bin = bin, cluster = cluster)
        on a weekday. bin 0 starts at start_min; each subsequent bin is +resolution_min.
        State indices follow OccupancyState: AWAY=0, HOME=1, SLEEP=2.
        Cluster indices are 0-based (cluster 0 = file cluster_01, etc.).

    weekend_transition_probs : dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]}
        Same as weekday_transition_probs but for weekends.
    """

    # Column order matches OccupancyState enum: AWAY=0, HOME=1, SLEEP=2
    _STATE_FROM_ORDER = {"Away": OccupancyState.AWAY.value, "Home": OccupancyState.HOME.value, "Sleep": OccupancyState.SLEEP.value}
    _TO_COLS = ["to_Away", "to_Home", "to_Sleep"]

    # Maps MobilityCluster string values to 0-based cluster indices
    CLUSTER_INDEX = {
        "mostly_home": 0,
        "long_day_away": 1,
        "morning_away": 2,
        "afternoon_away": 3,
        "evening_night_away": 4,
    }

    def __init__(self, num_clusters: int, start_min: int, assumption_resolution_min: int,
                 weekday_initial_probs: list[list[float]],
                 weekend_initial_probs: list[list[float]],
                 weekday_transition_probs: dict[int, list[list[list[float]]]],
                 weekend_transition_probs: dict[int, list[list[list[float]]]]):
        self.num_clusters = num_clusters
        self.start_min = start_min
        self.assumption_resolution_min = assumption_resolution_min
        # Pre-compute cumulative sums for fast searchsorted sampling
        # Shape per cluster: [num_bins, 3, 3] cumsum over last axis
        self.weekday_initial_probs = weekday_initial_probs
        self.weekend_initial_probs = weekend_initial_probs
        self.weekday_transition_probs = weekday_transition_probs
        self.weekend_transition_probs = weekend_transition_probs
        self._wd_init_cumsum = np.cumsum(np.array(weekday_initial_probs), axis=-1)
        self._we_init_cumsum = np.cumsum(np.array(weekend_initial_probs), axis=-1)
        self._wd_cumsum = {k: np.cumsum(np.array(v), axis=-1) for k, v in weekday_transition_probs.items()}
        self._we_cumsum = {k: np.cumsum(np.array(v), axis=-1) for k, v in weekend_transition_probs.items()}

    @staticmethod
    def _read_transition_probs(path) -> list[list[list[float]]]:
        """Read a time-varying transition CSV into a [num_bins, from_state, to_state] array."""
        df = pd.read_csv(path)
        df["from_idx"] = df["from_state"].map(ClusterAssumptions._STATE_FROM_ORDER)
        df = df.sort_values(["bin", "from_idx"])
        return [
            group[ClusterAssumptions._TO_COLS].values.tolist()
            for _, group in df.groupby("bin")
        ]

    @classmethod
    def default(cls) -> "ClusterAssumptions":
        data_dir = files("ob_generation.data.occupancy_probability")
        num_clusters = 5

        df = pd.read_csv(data_dir / "weekday" / "weekday_pi0.csv").sort_values("raw_cluster")
        weekday_initial_probs = df[["Away", "Home", "Sleep"]].values.tolist()

        df = pd.read_csv(data_dir / "weekend" / "weekend_pi0.csv").sort_values("raw_cluster")
        weekend_initial_probs = df[["Away", "Home", "Sleep"]].values.tolist()

        weekday_transition_probs = {
            i: cls._read_transition_probs(data_dir / "weekday" / f"cluster_{i+1:02d}_transitions.csv")
            for i in range(num_clusters)
        }
        weekend_transition_probs = {
            i: cls._read_transition_probs(data_dir / "weekend" / f"cluster_{i+1:02d}_transitions.csv")
            for i in range(num_clusters)
        }
        return ClusterAssumptions(num_clusters, 240, 15, weekday_initial_probs, weekend_initial_probs, weekday_transition_probs, weekend_transition_probs)
    
    def _build_cumsum(self, transition_matrices: np.ndarray, resolution_min: int) -> np.ndarray:
        """Build cumulative sum matrices, resampling from self.assumption_resolution_min to resolution_min.
        
        transition_matrices: shape (num_bins_at_assumption_res, num_states, num_states)
        Returns: cumsum of shape (num_bins_at_resolution_min, num_states, num_states)
        """
        if resolution_min == self.assumption_resolution_min:
            matrices = transition_matrices

        elif resolution_min < self.assumption_resolution_min:
            # Upsample: repeat each transition matrix (transition_resolution_min // resolution_min) times
            repeat = self.assumption_resolution_min // resolution_min
            matrices = np.repeat(transition_matrices, repeat, axis=0)

        else:
            # Downsample: compose consecutive transition matrices into coarser steps
            step = resolution_min // self.assumption_resolution_min
            num_coarse_bins = len(transition_matrices) // step
            matrices = np.empty((num_coarse_bins, transition_matrices.shape[1], transition_matrices.shape[2]))
            for b in range(num_coarse_bins):
                composed = transition_matrices[b * step]
                for k in range(1, step):
                    composed = composed @ transition_matrices[b * step + k]
                matrices[b] = composed

        return np.cumsum(matrices, axis=-1)
    


    def sample_cluster_annually(self, weekday_cluster_id: int, weekend_cluster_id: int, sim_resolution_min: int) -> list[list[OccupancyState]]:
        """Simulate a full year of occupancy states for a household in the given clusters.

        Returns a list of 53 weeks: 52 full weeks and a final partial week containing
        only Saturday. Each week is a flat list of OccupancyState values covering
        7 * (1440 // sim_resolution_min) time bins (or fewer for the last week), starting
        at Sunday midnight.

        The Markov Chain runs from 4am to 4am (next day). Since collection begins at
        Sunday midnight, a one-day pre-run is performed starting from Sunday 4am using
        the weekend transition matrix; the tail of that pre-run (midnight to 4am) seeds
        the first week. Collection then proceeds day-by-day for 365 days, using the
        weekend matrix for Sundays and Saturdays, and the weekday matrix otherwise.
        """
        num_bins = 1440 // sim_resolution_min
        midnight_bin = (1440 - self.start_min) // sim_resolution_min  
        we_cumsum = self._build_cumsum(self._we_cumsum[weekend_cluster_id], sim_resolution_min)
        wd_cumsum = self._build_cumsum(self._wd_cumsum[weekday_cluster_id], sim_resolution_min)

        # Pre-generate all random numbers in one call
        rng = np.random.default_rng()
        total_steps = 1 + 366 * num_bins  # 1 initial state + pre-run + 365 days
        randoms = rng.random(total_steps)
        r = 0

        # Sample one initial state at Sunday 4am (weekend)
        current_state = int(np.searchsorted(self._we_init_cumsum[weekend_cluster_id], randoms[r])); r += 1

        # Pre-run: run Sunday 4am -> Monday 4am to seed Sunday midnight->4am bins
        initial_state = []
        for b in range(num_bins):
            current_state = int(np.searchsorted(we_cumsum[b, current_state], randoms[r])); r += 1
            if b >= midnight_bin:
                initial_state.append(current_state)

        # Collect 365 days starting from Sunday midnight
        states = np.empty(366 * num_bins - midnight_bin, dtype=np.int8)
        states[:(num_bins - midnight_bin)] = initial_state
        idx = num_bins - midnight_bin

        for day in range(365):
            cumsum = we_cumsum if (day % 7) in [0, 6] else wd_cumsum  # day 0 = Sun, day 6 = Sat
            for b in range(num_bins):
                current_state = int(np.searchsorted(cumsum[b, current_state], randoms[r])); r += 1
                states[idx] = current_state; idx += 1

        flat = [OccupancyState(s) for s in states[:365 * num_bins]]
        bins_per_week = 7 * num_bins
        return [flat[i:i + bins_per_week] for i in range(0, len(flat), bins_per_week)]
  
    

class HouseholdOccupancyFractions(NamedTuple):
    home: float   # fraction of occupants in HOME state
    sleep: float  # fraction of occupants in SLEEP state
    # AWAY is implicit: 1.0 - home - sleep


class OccupancyGenerator:

    def __init__(self, occupancy: Occupancy, cluster_assumptions: ClusterAssumptions, sim_resolution_min: int):
        self.occupancy = occupancy
        self.cluster_assumptions = cluster_assumptions
        self.sim_resolution_min = sim_resolution_min




    def household_mc_state_annually(self) -> list[list[HouseholdOccupancyFractions]]:
        # 1. Simulate each occupant independently
        per_occupant = []
        for profile in self.occupancy.household_composition.occupants:
            wd_cluster = ClusterAssumptions.CLUSTER_INDEX[profile.weekday_cluster.value]
            we_cluster = ClusterAssumptions.CLUSTER_INDEX[profile.weekend_cluster.value]
            per_occupant.append(
                self.cluster_assumptions.sample_cluster_annually(wd_cluster, we_cluster, self.sim_resolution_min)
            )
        # per_occupant: shape (num_occupants, 53 weeks, bins_per_week)

        # 2. Aggregate per week
        weight = 1.0 / self.occupancy.num_occupants
        num_weeks = len(per_occupant[0])
        result = []
        for w in range(num_weeks):
            bins_per_week = len(per_occupant[0][w])
            counts = np.zeros((bins_per_week, 2), dtype=float)  # col 0=home, col 1=sleep
            for occupant_weeks in per_occupant:
                states_arr = np.array([s.value for s in occupant_weeks[w]], dtype=np.int8)
                counts[:, 0] += (states_arr == OccupancyState.HOME.value) * weight
                counts[:, 1] += (states_arr == OccupancyState.SLEEP.value) * weight
            result.append([HouseholdOccupancyFractions(home=counts[b, 0], sleep=counts[b, 1])
                        for b in range(bins_per_week)])
        return result
    
    def apply_away_time(self, occupancy_states: list[list[HouseholdOccupancyFractions]], away_time: list[None | TimeRange]) -> list[list[HouseholdOccupancyFractions]]:
        """Post-process the generated occupancy states to enforce away_time constraints.

        During away_time: force home=0, sleep=0 (all occupants away).
        Outside away_time: ensure home+sleep >= one_unit; if both are 0, set home=one_unit.
        """
        bins_per_day = 1440 // self.sim_resolution_min
        resolution_hours = self.sim_resolution_min / 60.0
        away = HouseholdOccupancyFractions(home=0.0, sleep=0.0)
        one_unit = 1.0 / self.occupancy.num_occupants

        for day, away_range in enumerate(away_time[:365]):
            week = day // 7
            day_start = (day % 7) * bins_per_day

            if away_range is None:
                for b in range(bins_per_day):
                    idx = day_start + b
                    f = occupancy_states[week][idx]
                    if f.home + f.sleep == 0.0:
                        occupancy_states[week][idx] = HouseholdOccupancyFractions(home=one_unit, sleep=0.0)
            else:
                for b in range(bins_per_day):
                    bin_hour = b * resolution_hours
                    idx = day_start + b
                    if away_range.contains_hour(bin_hour):
                        occupancy_states[week][idx] = away
                    else:
                        f = occupancy_states[week][idx]
                        if f.home + f.sleep == 0.0:
                            occupancy_states[week][idx] = HouseholdOccupancyFractions(home=one_unit, sleep=0.0)

        return occupancy_states
    
    def apply_sleep_time(self, occupancy_states: list[list[HouseholdOccupancyFractions]], sleep_time: list[None | TimeRange], max_awake_sleep_ratio: float = 0.0) -> list[list[HouseholdOccupancyFractions]]:
        """Post-process the generated occupancy states to enforce sleep_time constraints.

        During sleep_time: force sleep=1, home=0 (all occupants asleep).
        Outside sleep_time: clamp sleep ratio to max_awake_sleep_ratio.
        """
        bins_per_day = 1440 // self.sim_resolution_min
        resolution_hours = self.sim_resolution_min / 60.0
        asleep = HouseholdOccupancyFractions(home=0.0, sleep=1.0)

        for day, sleep_range in enumerate(sleep_time[:365]):
            if sleep_range is None:
                continue
            week = day // 7
            day_start = (day % 7) * bins_per_day
            for b in range(bins_per_day):
                idx = day_start + b
                if sleep_range.contains_hour(b * resolution_hours):
                    occupancy_states[week][idx] = asleep
                else:
                    f = occupancy_states[week][idx]
                    if f.sleep > max_awake_sleep_ratio:
                        excess = f.sleep - max_awake_sleep_ratio
                        occupancy_states[week][idx] = HouseholdOccupancyFractions(home=f.home + excess, sleep=max_awake_sleep_ratio)

        return occupancy_states



    def household_away_time_annually(self) -> list[None | TimeRange]:
        """Generate a year's worth of away/home occupancy for the household.
            If specified in the occupancy patterns, the away_time will be forced to be away during the sampled away_interval given the rigidness. 
            This should only be called if the household has a specified away_interval in either weekday or weekend pattern. 
        """
        no_weekday_away = self.occupancy.weekday_pattern.is_always_occupied

        if not no_weekday_away:
            weekday_var = self.occupancy.weekday_pattern.away_time_rigidness.to_std_dev_hours() if self.occupancy.weekday_pattern.away_time_rigidness else 1.0
            weekday_away_time_distribution = TimeRangeDistribution(
                time_range=self.occupancy.weekday_pattern.away_interval,
                start_variance= weekday_var,
                end_variance= weekday_var,
                resolution_mins=self.sim_resolution_min
            )

        no_weekend_away = self.occupancy.weekend_pattern.is_always_occupied
        if not no_weekend_away:
            weekend_var = self.occupancy.weekend_pattern.away_time_rigidness.to_std_dev_hours() if self.occupancy.weekend_pattern.away_time_rigidness else 1.0
            weekend_away_time_distribution = TimeRangeDistribution(
                time_range=self.occupancy.weekend_pattern.away_interval,
                start_variance= weekend_var,
                end_variance= weekend_var,
                resolution_mins=self.sim_resolution_min
            )
            
        #355 length list of daily away_time (None if no away time that day, else the TimeRange for that day)
        states = np.empty(365 , dtype=object)  # Will hold TimeRange or None for each day
        for day in range(365):
            if (day % 7) in [0, 6]:  # Weekend
                if no_weekend_away:
                    states[day] = None 
                else:
                    away_time_interval = weekend_away_time_distribution.sample()
                    states[day] = away_time_interval
            else:  # Weekday
                if no_weekday_away:
                    states[day] = None
                else:
                    states[day] = weekday_away_time_distribution.sample()

        return states.tolist()
    
    def household_sleep_time_annually(self, away_time: list[None | TimeRange]) -> list[None | TimeRange]:
        """Generate a year's worth of sleep/awake occupancy for the household.
            If specified in the occupancy patterns, the sleep_time will be forced to be sleep during the sampled sleep_time given the rigidness. 
            This should only be called if the household has a specified sleep_time in the sleep pattern. 
        """
        no_sleep = self.occupancy.sleep_pattern.is_always_awake
        if no_sleep:
            return [None] * 365
        
        sleep_var = self.occupancy.sleep_pattern.sleep_time_rigidness.to_std_dev_hours() if self.occupancy.sleep_pattern.sleep_time_rigidness else 1.0
        sleep_time_distribution = TimeRangeDistribution(
            time_range=self.occupancy.sleep_pattern.sleep_time,
            start_variance= sleep_var,
            end_variance= sleep_var,
            resolution_mins=self.sim_resolution_min
        )
        states = np.empty(365, dtype=object)  # Will hold TimeRange or None for each day
        for day in range(365):
            sleep_range = sleep_time_distribution.sample_with_away_bounds(away_time[day])
            states[day] = sleep_range

        return states.tolist()
        
 
    def generate(self) -> list[list[HouseholdOccupancyFractions]]:
        """Main method to generate occupancy states for the household.
            1. Generate initial occupancy states using Markov Chain based on the mobility clusters of the occupants. This will give us a list of 53 weeks, each week is a list of HouseholdOccupancyFractions for each time bin in that week. 
            (2,3 steps only process the generated occupancy states if the corresponding patterns are specified in the occupancy input)
            2. If available, Generate away_time and sleep_time for each day using the specified patterns and rigidness. This will give us two lists of 365 elements (one per day), where each element is either None (if no away/sleep time that day) or a TimeRange specifying the away/sleep interval for that day.
            3. Post-process the generated occupancy states to enforce the away_time and sleep_time constraints. For bins that fall within an away_time, set occupancy to fully away; for bins that fall within a sleep_time, set occupancy to fully sleep.
        """
        occ_states = self.household_mc_state_annually()
        occupancy = self.occupancy
        away_time = [None] * 365
        if occupancy.weekday_pattern is not None or occupancy.weekend_pattern is not None:
            if occupancy.weekend_pattern is  None:
                self.occupancy.weekend_pattern = WeekendOccupancyPattern(is_always_occupied=True)
            if occupancy.weekday_pattern is None:
                self.occupancy.weekday_pattern = WeekdayOccupancyPattern(is_always_occupied=True)
            away_time = self.household_away_time_annually()
            occ_states = self.apply_away_time(occ_states, away_time)
        if occupancy.sleep_pattern is not None:
            sleep_time = self.household_sleep_time_annually(away_time)
            occ_states = self.apply_sleep_time(occ_states, sleep_time)
        return occ_states
    
    @staticmethod
    def active_sleep_mask(occ_states:list[list[HouseholdOccupancyFractions]], active_threshold: float = 0.3) -> tuple[list[list[bool]], list[list[bool]]]:
        """Given the generated occupancy states, produce binary masks for active and sleep states based on the specified active_threshold.

        A bin is considered "active" if the home fraction >= active_threshold.
        A bin is considered "sleep" if the sleep fraction >= active_threshold.
        """
        active_mask = [[False] * len(week) for week in occ_states]
        sleep_mask = [[False] * len(week) for week in occ_states]
        for w in range(len(occ_states)):
            week = occ_states[w]
            for t in range(len(week)):
                time_bin = week[t]
                total_home_sleep = time_bin.home + time_bin.sleep
                if total_home_sleep > 0.0: 
                    ratio = time_bin.home / total_home_sleep
                    if ratio >= active_threshold:
                        active_mask[w][t] = True
                    else:
                        sleep_mask[w][t] = True
        return active_mask, sleep_mask
    
    @staticmethod
    def apply_to_mask(mask:list[bool] , existing_schedule:list[float], value:float) -> list[float]:
        """ Apply a binary mask to an existing schedule, setting masked bins to the specified value while leaving unmasked bins unchanged."""
        assert len(mask) == len(existing_schedule), "mask and schedule must have the same length"
        return [
            value if masked else v
            for v, masked in zip(existing_schedule, mask)
        ]
    
    @staticmethod
    def to_occupancy_schedule(occ_states: list[list[HouseholdOccupancyFractions]]) -> list[float]:
        """Convert the generated occupancy states into a schedule of home occupancy fractions for each time bin across the year.

        This flattens the weekly structure into a single list of 365 * bins_per_day fractions, where each fraction represents the expected proportion of occupants at home (including both active and sleep) during that time bin.
        """
        schedule = []
        for week in occ_states:
            for time_bin in week:
                schedule.append(time_bin.home + time_bin.sleep)
        return schedule