from stochastic.distribution import Distribution,NormalDistribution
from model.occupancy import Occupancy, TimeRange
from stochastic.distribution_config import DistributionConfig
from enum import Enum
import random
from pydantic import BaseModel, Field, ConfigDict
import json
from pathlib import Path
from typing import Optional


class OccupantRole(Enum):
    DAILY_COMMUTER = "daily_commuter"
    HYBRID_WORKER = "hybrid_worker"
    STAYATHOME = "stayathome"
    K12_OR_DAYCARE = "k12_or_daycare"
    COLLEGE_STUDENT = "college_student"

class RoleAssumption(BaseModel):
    """Defines the behavior for a specific occupant role."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: OccupantRole = Field(..., description="The occupant role.")
    days_away_freq: Distribution = Field(..., description="How many days/week this role leaves home.")
    leave_time: Optional[Distribution] = Field(..., description="Time of departure.")
    return_time: Optional[Distribution] = Field(..., description="Time of return.")

class OccupancyAssumptions(BaseModel):
    """Master configuration for household behavioral assumptions."""
    daily_commuter: RoleAssumption
    hybrid_worker: RoleAssumption
    stayathome: RoleAssumption
    k12_or_daycare: RoleAssumption
    college_student: RoleAssumption

    @staticmethod
    def _build_role(d: dict) -> RoleAssumption:
        return RoleAssumption(
            role=OccupantRole(d["role"]),
            days_away_freq=DistributionConfig(**d["days_away_freq"]).build(),
            leave_time=DistributionConfig(**d["leave_time"]).build(),
            return_time=DistributionConfig(**d["return_time"]).build(),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "OccupancyAssumptions":
        data = json.loads(Path(path).read_text())

        return cls(
            daily_commuter=cls._build_role(data["daily_commuter"]),
            hybrid_worker=cls._build_role(data["hybrid_worker"]),
            stayathome=cls._build_role(data["stayathome"]),
            k12_or_daycare=cls._build_role(data["k12_or_daycare"]),
            college_student=cls._build_role(data["college_student"]),
        )

    @classmethod
    def default(cls) -> "OccupancyAssumptions":
        """Returns the standard/default assumptions."""
        return cls(
            daily_commuter=RoleAssumption(
                role = OccupantRole.DAILY_COMMUTER,
                days_away_freq=DistributionConfig(dist_type="constant", params={"value": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 1.3, "lower": 0.0, "upper": 24.0, "int": False}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 1.3, "lower": 0.0, "upper": 24.0, "int": False}).build()
            ),
            hybrid_worker=RoleAssumption(
                role = OccupantRole.HYBRID_WORKER,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 1, "max": 4}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 1.3, "lower": 0.0, "upper": 24.0, "int": False}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 1.3, "lower": 0.0, "upper": 24.0, "int": False}).build()
            ),
            stayathome=RoleAssumption(
                role = OccupantRole.STAYATHOME,
                days_away_freq=DistributionConfig(dist_type="constant", params={"value": 0}).build(),
                leave_time= None,
                return_time= None
            ),
            k12_or_daycare=RoleAssumption(
                role = OccupantRole.K12_OR_DAYCARE,
                days_away_freq=DistributionConfig(dist_type="constant", params={"value": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 7.5, "std": 1.0, "lower": 0.0, "upper": 24.0, "int": False}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 15.0, "std": 1.0, "lower": 0.0, "upper": 24.0, "int": False}).build()
            ),
            college_student=RoleAssumption(
                role = OccupantRole.COLLEGE_STUDENT,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 3, "max": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 9.0, "std": 1.5, "lower": 0.0, "upper": 24.0, "int": False}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 3.0, "lower": 0.0, "upper": 24.0, "int": False}).build()
            )
        )
    



class TimeRangeDistribution:
    """Distribution for time ranges within a day."""
    def __init__(self, time_range: TimeRange, start_variance:float, end_variance:float, resolution_mins: int = 15):
        self.resolution_mins = resolution_mins
        self.resolution_hours = resolution_mins / 60
        upper_bound = 24.0 - self.resolution_hours
        self.start_dist = NormalDistribution(
            mean= time_range.start_hour ,
            stddev=start_variance,
            lower=0.0,
            upper=upper_bound,
            int=False
        )
        self.end_dist = NormalDistribution(
            mean= time_range.end_hour ,
            stddev=end_variance,
            lower=0.0,
            upper=upper_bound,
            int=False
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

    @staticmethod
    def sample_with_bounds(start_dist: NormalDistribution, end_dist: NormalDistribution, away_time: Optional[TimeRange], resolution_hours: float) -> Optional[TimeRange]:
        """
        Samples a single occupant's leave and return times ensuring logical consistency.
        Leave and return times must be outside the away_time period (when no one is home).

        Args:
            start_dist: Distribution for leave time
            end_dist: Distribution for return time
            away_time: TimeRange defining when no one is home
        Returns:
            Tuple of (start_time, end_time)
        """
        if away_time is None:
            updated_start_dist = start_dist
            updated_end_dist = end_dist
        elif away_time.wraps_midnight:
            updated_start_dist = start_dist.with_bounds(lower=away_time.end_hour, upper=away_time.start_hour)
            updated_end_dist = end_dist.with_bounds(lower=away_time.end_hour, upper=away_time.start_hour)
        else:
            updated_start_dist = start_dist.with_bounds(lower=0, upper=away_time.start_hour)
            updated_end_dist = end_dist.with_bounds(lower=away_time.end_hour, upper=24 - resolution_hours)

        MAX_ATTEMPTS = 1000
        INNER_ATTEMPTS = 20

        total_attempts = 0
        while total_attempts < MAX_ATTEMPTS:
            start_time = TimeRangeDistribution._snap_to_resolution(updated_start_dist.sample(), resolution_hours=resolution_hours)
            for _ in range(INNER_ATTEMPTS):
                end_time = TimeRangeDistribution._snap_to_resolution(updated_end_dist.sample(), resolution_hours=resolution_hours)
                if away_time is None or away_time.wraps_midnight or end_time > start_time:
                    return TimeRange(start_hour=start_time, end_hour=end_time)
            total_attempts += INNER_ATTEMPTS

        # Fallback: if bounds make it impossible to satisfy end > start,
        if away_time is None:
            return TimeRange(start_hour=start_time, end_hour=end_time)
        else:
            return away_time



class SingleOccupantTracker:
    def __init__(self, role: OccupantRole, assumption: RoleAssumption, resolution_mins):
        self.role = role
        self.assumption = assumption
        self.resolution_hours = resolution_mins / 60

    def sample_weekdays(self, household_away_intervals: list[TimeRange | None], away_days: list[int]) -> list[TimeRange | None]:
        """Samples the weekly leave and return times for this occupant based on their role and household patterns"""
        intervals = []
        if self.role == OccupantRole.DAILY_COMMUTER or self.role == OccupantRole.K12_OR_DAYCARE:
            for d in range(5):
                time_range = TimeRangeDistribution.sample_with_bounds(
                    self.assumption.leave_time,
                    self.assumption.return_time,
                    household_away_intervals[d],
                    self.resolution_hours
                )
                intervals.append(time_range)
        elif self.role == OccupantRole.HYBRID_WORKER or self.role == OccupantRole.COLLEGE_STUDENT:
            # if household level always occupied, then sample for away on site days, else use household away days for away on site days
            if not away_days:
                away_days = random.sample(range(5), self.assumption.days_away_freq.sample())
            # hybrid worker cannot have all 5 days following onsite schedule
            elif self.role == OccupantRole.HYBRID_WORKER and len(away_days) == 5:
                away_days = random.sample(range(5), self.assumption.days_away_freq.sample())
            for d in range(5):
                if d in away_days:
                    time_range = TimeRangeDistribution.sample_with_bounds(
                        self.assumption.leave_time,
                        self.assumption.return_time,
                        household_away_intervals[d],
                        self.resolution_hours
                    )
                    intervals.append(time_range)
                else:
                    intervals.append(household_away_intervals[d])
        elif self.role == OccupantRole.STAYATHOME:
            intervals = household_away_intervals
        return intervals
       


class OccupancyGenerator:

    def __init__(self, occupancy: Occupancy, assumptions: OccupancyAssumptions, resolution_mins: int = 15):
        if 60 % resolution_mins != 0:
            raise ValueError(
                f"resolution_mins must evenly divide 60. Got {resolution_mins}. "
                f"Valid values: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60"
            )
        self.occupancy = occupancy
        self.assumptions = assumptions
        self.occupants_cnt = occupancy.num_occupants
        if self.occupancy.weekday_pattern.is_always_occupied:
            self.weekday_away = None
        else:
            self.weekday_away = TimeRangeDistribution(
                occupancy.weekday_pattern.away_pattern.away_interval,
                start_variance=1.0,
                end_variance=1.0,
                resolution_mins=resolution_mins) if occupancy.weekday_pattern.away_pattern else None
        if self.occupancy.weekend_pattern.is_always_occupied:
            self.weekend_away = None
        else:
            self.weekend_away = TimeRangeDistribution(
                occupancy.weekend_pattern.away_pattern.away_interval,
                start_variance=2.0,
                end_variance=2.0,
                resolution_mins=resolution_mins) if occupancy.weekend_pattern.away_pattern else None
        self.trackers = self._get_single_occupant_tracker(resolution_mins)
        self.num_per_hour = 60 // resolution_mins
        self.resolution_mins = resolution_mins
        
            
    def _get_single_occupant_tracker(self, resolution_mins: int) -> list[SingleOccupantTracker]:
        """Generates a list of SingleOccupantTracker instances based on the household composition."""
        trackers = []
        comp = self.occupancy.household_composition
        role_counts = {
            OccupantRole.DAILY_COMMUTER: comp.daily_commuter,
            OccupantRole.HYBRID_WORKER: comp.hybrid_worker,
            OccupantRole.STAYATHOME: comp.stayathome,
            OccupantRole.K12_OR_DAYCARE: comp.k12_or_daycare,
            OccupantRole.COLLEGE_STUDENT: comp.college_student
        }
        for role, count in role_counts.items():
            assumption = getattr(self.assumptions, role.value)
            for _ in range(count):
                trackers.append(SingleOccupantTracker(role, assumption, resolution_mins))
        return trackers
        
    def weekday_away_interval(self) -> tuple[list[TimeRange | None], list[int]]:
        """ Based on the household's weekday away patterns, sample and return the away intervals for each weekday."""
        occupancy = self.occupancy
        if occupancy.weekday_pattern.is_always_occupied:
            return [None] * 5, []  # No one leaves on weekdays
        else:
            weekdays = [0,1,2,3,4]
            away_days = random.sample(weekdays, occupancy.weekday_pattern.away_pattern.num_of_days)
            interval = []
            for d in weekdays:
                if d in away_days:
                    interval.append(self.weekday_away.sample())
                else:
                    interval.append(None)
            return interval,away_days
        
    def household_weekday_schedule(self) -> list[float]:
        """Generates the household's overall weekday away schedule based on individual occupant patterns."""
        weekday_away_intervals, away_days = self.weekday_away_interval()
        all_occupant_intervals = []
        for tracker in self.trackers:
            occupant_intervals = tracker.sample_weekdays(weekday_away_intervals, away_days)
            all_occupant_intervals.append(occupant_intervals)

        schedule = []
        ratio = 1.0 / self.occupants_cnt 

        for d in range(5):
            away_interval = weekday_away_intervals[d]
            for h in range(24 * self.num_per_hour):
                hour = h / self.num_per_hour
                # --- REGION 1: ABSOLUTE ZERO (Hard Constraint) ---
                if away_interval is not None:
                    if away_interval.contains_hour(hour):
                        schedule.append(0.0)
                        continue 

                # -- REGION 2: CALCULATE RAW PRESENCE ---
                people_present = 0
                for i in range(self.occupants_cnt):
                    occupant_away_interval = all_occupant_intervals[i][d]
                    if occupant_away_interval is None or not occupant_away_interval.contains_hour(hour):
                       people_present += 1

                # -- REGION 3: LAST MAN STANDING (The Fix) ---
                # If no one is calculated to be present but we're outside the 
                # absolute zero range, ensure at least one person is home
                if people_present == 0:
                    people_present = 1
                
                # Calculate fraction
                frac = round(people_present * ratio, 4)
                schedule.append(frac)
        return schedule
    

    def household_weekend_schedule(self) -> list[float]:
        """Generates the household's overall weekend away schedule based on individual occupant patterns."""
        occupancy = self.occupancy
        if occupancy.weekend_pattern.is_always_occupied:
            return [1.0] * (2 * 24 * self.num_per_hour)
        else:
            schedule = []
            for _ in range(2):
                all_occupant_intervals = []
                for _ in range(self.occupants_cnt):
                    interval = self.weekend_away.sample()
                    all_occupant_intervals.append(interval)
                day_schedule = []
                for h in range(24 * self.num_per_hour):
                    hour = h / self.num_per_hour
                    people_present = 0
                    for i in range(self.occupants_cnt):
                        occupant_away_interval = all_occupant_intervals[i]
                        if not occupant_away_interval.contains_hour(hour):
                            people_present += 1
                    frac = round(people_present / self.occupants_cnt, 4)
                    day_schedule.append(frac)
                schedule.extend(day_schedule)
        return schedule
    
    def household_fullweek_schedule(self) -> list[float]:
        """Generates the household's full week away schedule based on individual occupant patterns."""
        weekday_schedule = self.household_weekday_schedule()
        weekend_schedule = self.household_weekend_schedule()
        fullweek_schedule = weekday_schedule + weekend_schedule
        return fullweek_schedule
    
    def household_sleep_schedule(self) -> list[TimeRange | None]:
        """Generates the household's sleep time schedule."""
        sleep_time = self.occupancy.sleep_time
        if sleep_time is None:
            return [None] * 7
        else:
            sleep_range_dist = TimeRangeDistribution(
                sleep_time,
                start_variance=1.0,
                end_variance=1.0,
                resolution_mins= self.resolution_mins
            )
            sleep_schedule = []
            for _ in range(7):
                sleep_schedule.append(sleep_range_dist.sample())
            return sleep_schedule
    
    @staticmethod
    def get_sleep_mask(sleep_schedule: list[Occupancy.TimeRange | None], num_per_hour: int) -> list[bool]:
        """ Generates a mask indicating occupied and sleep hours (True) vs unoccupied or active hours (False)."""
        mask = [False] * (24 * 7 * num_per_hour)
        for d in range(7):
            sleep_info = sleep_schedule[d]
            if sleep_info is not None:
                for h in range(24 * num_per_hour):
                    hour = h / num_per_hour
                    hour_index = d * 24 * num_per_hour + h
                    if sleep_info.contains_hour(hour):
                        mask[hour_index] = True
        return mask
    
    @staticmethod
    def revise_by_sleep(sleep_weekly_mask:list[bool] , existing_schedule:list[float], value:float) -> list[float]:
        """ Revisions to an existing schedule based on sleep times. 
            Assume both inputs (sleep_weekly_mask and existing_schedule) should be in the same length."""
        assert len(sleep_weekly_mask) == len(existing_schedule), "sleep mask and schedule must have the same length"
        return [
            value if asleep else v
            for v, asleep in zip(existing_schedule, sleep_weekly_mask)
        ]
    
    @staticmethod
    def get_occupancy_mask(occupancy_schedule: list[float]) -> list[bool]:
        """ Generates a mask indicating occupied hours (True) vs unoccupied hours (False)."""
        mask = [occ > 0.0 for occ in occupancy_schedule]
        return mask
    
    @staticmethod
    def revise_by_absence(occupancy_weekly_mask:list[bool] , existing_schedule:list[float], value:float) -> list[float]:
        """ Revisions to an existing schedule based on occupancy times. 
            Assume both inputs (occupancy_weekly_mask and existing_schedule) should be in the same length."""
        assert len(occupancy_weekly_mask) == len(existing_schedule), "occupancy mask and schedule must have the same length"
        return [
            value if not occupied else v
            for v, occupied in zip(existing_schedule, occupancy_weekly_mask)
        ]
    

    def household_annual_schedule(self) -> tuple[list[list[float]], list[list[bool]]]:
        """Generates the household's annual schedule by repeating the weekly schedule 52 weeks + 1 day."""
        annual_schedule = []
        annual_sleep_schedule = []
        for w in range(53):
            weekly_schedule = self.household_fullweek_schedule()
            sleep_schedule = self.household_sleep_schedule()
            sleep_mask = OccupancyGenerator.get_sleep_mask(sleep_schedule, self.num_per_hour)
            if w < 52:
                annual_schedule.append(weekly_schedule)
                annual_sleep_schedule.append(sleep_mask)
            else:
                # For the 53rd week, only add the first day (to make 365 days)
                annual_schedule.append(weekly_schedule[:24 * self.num_per_hour])
                annual_sleep_schedule.append(sleep_mask[:24 * self.num_per_hour])
        return annual_schedule, annual_sleep_schedule
    
        
            

                
        
