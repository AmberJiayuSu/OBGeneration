from obgeneration.stochastic.markov import ClusterAssumptions, OccupancyState
from obgeneration.stochastic.time_range import TimeRangeDistribution
from obgeneration.model.occupancy import Occupancy, TimeRange, WeekendOccupancyPattern, WeekdayOccupancyPattern
from obgeneration.generator.results import OccupancyResult
from obgeneration.generator.types import HouseholdOccupancyFractions
import math
import numpy as np
from collections.abc import Sequence
import copy


class OccupancyGenerator:

    def __init__(self, occupancy: Occupancy, cluster_assumptions: ClusterAssumptions, sim_resolution_min: int):
        self.occupancy = occupancy
        self.cluster_assumptions = cluster_assumptions
        self.sim_resolution_min = sim_resolution_min

    @staticmethod
    def generate_result(
        occupancy: Occupancy,
        cluster_assumptions: ClusterAssumptions,
        resolution_mins: int,
        rng: np.random.Generator | int,
    ) -> OccupancyResult:
        """Generate an annual occupancy result with explicit assumptions."""
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        generator = OccupancyGenerator(
            occupancy=occupancy,
            cluster_assumptions=cluster_assumptions,
            sim_resolution_min=resolution_mins,
        )
        occ_states = generator.generate(rng)
        return OccupancyResult(
            peak_value=float(occupancy.num_occupants),
            schedule=OccupancyGenerator.to_occupancy_schedule(occ_states),
            occupancy_states=occ_states,
        )

    @staticmethod
    def generate_with_defaults(
        occupancy: Occupancy,
        resolution_mins: int,
        rng: np.random.Generator | int,
    ) -> OccupancyResult:
        """Generate an annual occupancy result using default assumptions."""
        return OccupancyGenerator.generate_result(
            occupancy=occupancy,
            cluster_assumptions=ClusterAssumptions.default(),
            resolution_mins=resolution_mins,
            rng=rng,
        )


    def household_mc_state_annually(self, rng: np.random.Generator) -> list[list[HouseholdOccupancyFractions]]:
        # 1. Simulate each occupant independently
        per_occupant = []
        for profile in self.occupancy.household_composition.occupants:
            wd_cluster = ClusterAssumptions.CLUSTER_INDEX[profile.weekday_cluster.value]
            we_cluster = ClusterAssumptions.CLUSTER_INDEX[profile.weekend_cluster.value]
            per_occupant.append(
                self.cluster_assumptions.sample_cluster_annually(wd_cluster, we_cluster, self.sim_resolution_min, rng)
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
    
    def apply_away_time(self, occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]], away_time: Sequence[None | TimeRange]) -> list[list[HouseholdOccupancyFractions]]:
        """Post-process the generated occupancy states to enforce away_time constraints.

        During away_time: force home=0, sleep=0 (all occupants away).
        Outside away_time: ensure home+sleep >= one_unit; if both are 0, set home=one_unit.
        """
        bins_per_day = 1440 // self.sim_resolution_min
        resolution_hours = self.sim_resolution_min / 60.0
        away = HouseholdOccupancyFractions(home=0.0, sleep=0.0)
        one_unit = 1.0 / self.occupancy.num_occupants

        new_states = copy.deepcopy(occupancy_states)

        for day, away_range in enumerate(away_time[:365]):
            week = day // 7
            day_start = (day % 7) * bins_per_day

            if away_range is None:
                for b in range(bins_per_day):
                    idx = day_start + b
                    f = new_states[week][idx]
                    if f.home + f.sleep == 0.0:
                        new_states[week][idx] = HouseholdOccupancyFractions(home=one_unit, sleep=0.0)
            else:
                for b in range(bins_per_day):
                    bin_hour = b * resolution_hours
                    idx = day_start + b
                    if away_range.contains_hour(bin_hour):
                        new_states[week][idx] = away
                    else:
                        f = new_states[week][idx]
                        if f.home + f.sleep == 0.0:
                            new_states[week][idx] = HouseholdOccupancyFractions(home=one_unit, sleep=0.0)

        return new_states

    def apply_sleep_time(self, occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]], sleep_time: Sequence[None | TimeRange], active_threshold: float = 0.3) -> list[list[HouseholdOccupancyFractions]]:
        """Post-process the generated occupancy states to enforce sleep_time constraints.

        During sleep_time: force sleep=1, home=0 (all occupants asleep).
        Outside sleep_time: clamp sleep ratio to at least one person awake
        """
        bins_per_day = 1440 // self.sim_resolution_min
        resolution_hours = self.sim_resolution_min / 60.0
        asleep = HouseholdOccupancyFractions(home=0.0, sleep=1.0)
        # round up to the nearest multiples of 1 / num_occupants
        active_threshold = math.ceil(active_threshold * self.occupancy.num_occupants) / self.occupancy.num_occupants

        new_states = copy.deepcopy(occupancy_states)

        for day, sleep_range in enumerate(sleep_time[:365]):
            if sleep_range is None:
                continue
            week = day // 7
            day_start = (day % 7) * bins_per_day
            for b in range(bins_per_day):
                idx = day_start + b
                if sleep_range.contains_hour(b * resolution_hours):
                    new_states[week][idx] = asleep
                else:
                    f = new_states[week][idx]
                    #no one home
                    if (f.home + f.sleep)== 0.0:
                        continue
                    else:
                        active_ratio = f.home / (f.home + f.sleep) 
                        if active_ratio < active_threshold:
                            excess = active_threshold - active_ratio
                            new_states[week][idx] = HouseholdOccupancyFractions(home=f.home + excess, sleep= f.sleep - excess)

        return new_states



    def household_away_time_annually(self, rng: np.random.Generator) -> list[None | TimeRange]:
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
                    away_time_interval = weekend_away_time_distribution.sample(rng)
                    states[day] = away_time_interval
            else:  # Weekday
                if no_weekday_away:
                    states[day] = None
                else:
                    states[day] = weekday_away_time_distribution.sample(rng)

        return states.tolist()
    

    def household_sleep_time_annually(self, rng: np.random.Generator, away_time: Sequence[None | TimeRange]) -> list[None | TimeRange]:
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
            sleep_range = sleep_time_distribution.sample_with_away_bounds(rng, away_time[day])
            states[day] = sleep_range

        return states.tolist()
        
 
    def generate(self, rng: np.random.Generator) -> list[list[HouseholdOccupancyFractions]]:
        """Main method to generate occupancy states for the household.
            1. Generate initial occupancy states using Markov Chain based on the mobility clusters of the occupants. This will give us a list of 53 weeks, each week is a list of HouseholdOccupancyFractions for each time bin in that week.
            (2,3 steps only process the generated occupancy states if the corresponding patterns are specified in the occupancy input)
            2. If available, Generate away_time and sleep_time for each day using the specified patterns and rigidness. This will give us two lists of 365 elements (one per day), where each element is either None (if no away/sleep time that day) or a TimeRange specifying the away/sleep interval for that day.
            3. Post-process the generated occupancy states to enforce the away_time and sleep_time constraints. For bins that fall within an away_time, set occupancy to fully away; for bins that fall within a sleep_time, set occupancy to fully sleep.
        """
        occ_states = self.household_mc_state_annually(rng)
        occupancy = self.occupancy
        away_time = [None] * 365
        if occupancy.weekday_pattern is not None or occupancy.weekend_pattern is not None:
            if occupancy.weekend_pattern is None:
                self.occupancy.weekend_pattern = WeekendOccupancyPattern(is_always_occupied=True)
            if occupancy.weekday_pattern is None:
                self.occupancy.weekday_pattern = WeekdayOccupancyPattern(is_always_occupied=True)
            away_time = self.household_away_time_annually(rng)
            occ_states = self.apply_away_time(occ_states, away_time)
        if occupancy.sleep_pattern is not None:
            sleep_time = self.household_sleep_time_annually(rng, away_time)
            occ_states = self.apply_sleep_time(occ_states, sleep_time)
        return occ_states
    
    @staticmethod
    def active_sleep_mask(occ_states: Sequence[Sequence[HouseholdOccupancyFractions]], active_threshold: float = 0.3) -> tuple[list[list[bool]], list[list[bool]]]:
        """Given the generated occupancy states, produce binary masks for active and sleep states based on the specified active_threshold.

        A bin is considered "active" if the home fraction >= active_threshold.
        A bin is considered "sleep" otherwise.
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
    def apply_to_mask(mask: Sequence[bool], existing_schedule: Sequence[float], value: float) -> list[float]:
        """ Apply a binary mask to an existing schedule, setting masked bins to the specified value while leaving unmasked bins unchanged."""
        assert len(mask) == len(existing_schedule), "mask and schedule must have the same length"
        return [
            value if masked else v
            for v, masked in zip(existing_schedule, mask)
        ]
    
    @staticmethod
    def to_occupancy_schedule(occ_states: Sequence[Sequence[HouseholdOccupancyFractions]]) -> list[float]:
        """Convert the generated occupancy states into a schedule of home occupancy fractions for each time bin across the year.

        This flattens the weekly structure into a single list of 365 * bins_per_day fractions, where each fraction represents the expected proportion of occupants at home (including both active and sleep) during that time bin.
        """
        schedule = []
        for week in occ_states:
            for time_bin in week:
                schedule.append(time_bin.home + time_bin.sleep)
        return schedule
