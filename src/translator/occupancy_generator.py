from stochastic.distribution import Distribution,NormalDistribution
from model.occupancy import Occupancy, TimeRange
from stochastic.distribution_config import DistributionConfig
from enum import Enum
import random
from pydantic import BaseModel, Field, ConfigDict
import json
from pathlib import Path



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
    leave_time: Distribution = Field(..., description="Time of departure.")
    return_time: Distribution = Field(..., description="Time of return.")

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
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 0.5, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 0.5, "int": True}).build()
            ),
            hybrid_worker=RoleAssumption(
                role = OccupantRole.HYBRID_WORKER,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 1, "max": 4}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 0.5, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 0.5, "int": True}).build()
            ),
            stayathome=RoleAssumption(
                role = OccupantRole.STAYATHOME,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 0, "max": 2}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 11.0, "std": 2.0, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 13.0, "std": 2.0, "int": True}).build()
            ),
            k12_or_daycare=RoleAssumption(
                role = OccupantRole.K12_OR_DAYCARE,
                days_away_freq=DistributionConfig(dist_type="constant", params={"value": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 0.5, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 15.5, "std": 0.5, "int": True}).build()
            ),
            college_student=RoleAssumption(
                role = OccupantRole.COLLEGE_STUDENT,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 3, "max": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 10.0, "std": 3.0, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 16.0, "std": 3.0, "int": True}).build()
            )
        )
    



class TimeRangeDistribution:
    """Distribution for time ranges within a day."""
    def __init__(self, time_range: TimeRange, start_variance:float, end_variance:float):
        self.start_dist = NormalDistribution(
            mean= time_range.start_hour ,
            stddev=start_variance,
            lower=0,
            upper=23,
            int=True
        )
        self.end_dist = NormalDistribution(
            mean= time_range.end_hour ,
            stddev=end_variance,
            lower=0,
            upper=23,
            int=True
        )
        self.wraps_midnight = time_range.wraps_midnight

    
    def sample(self) -> tuple[int, int]:
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
        start_time = self.start_dist.sample()

        for _ in range(MAX_ATTEMPTS):
            end_time = self.end_dist.sample()
            if self.wraps_midnight or end_time > start_time:
                return start_time, end_time

        # Fallback: if bounds make it impossible to satisfy end > start,
        # allow end >= start to avoid infinite loop
        return start_time, end_time   


class OccupancyGenerator:

    def __init__(self, occupancy: Occupancy, assumptions: OccupancyAssumptions):
        self.occupancy = occupancy
        self.assumptions = assumptions
        self.occupants_cnt = occupancy.num_occupants
        if self.occupancy.weekday_pattern.is_always_occupied:
            self.primary_weekday_away = None
            self.secondary_weekday_away = None
        else:
            self.primary_weekday_away = TimeRangeDistribution(
                occupancy.weekday_pattern.primary_away_interval.away_interval,
                start_variance=1.0,
                end_variance=1.0) if occupancy.weekday_pattern.primary_away_interval else None
            self.secondary_weekday_away = TimeRangeDistribution(
                occupancy.weekday_pattern.secondary_away_interval.away_interval,
                start_variance=1.0,
                end_variance=1.0) if occupancy.weekday_pattern.secondary_away_interval else None
            
        
    def weekly_away_interval(self) -> list[TimeRange | None]:
        """ Based on the household's weekday away patterns, sample and return the away intervals for each weekday."""
        occupancy = self.occupancy
        if occupancy.weekday_pattern.is_always_occupied:
            return [None] * 5  # No one leaves on weekdays
        else:
            weekdays = [1, 2, 3, 4, 5]
            primary_days = random.sample(weekdays, occupancy.weekday_pattern.primary_away_interval.num_of_days)
            remaining_days = [d for d in weekdays if d not in primary_days]
            secondary_days = []
            if occupancy.weekday_pattern.secondary_away_interval:
                secondary_days = random.sample(remaining_days, occupancy.weekday_pattern.secondary_away_interval.num_of_days)
            remaining_days = [d for d in remaining_days if d not in secondary_days]

            interval = []
            for d in weekdays:
                if d in primary_days:
                    start, end = self.primary_weekday_away.sample()
                    interval.append(TimeRange(start_hour=start, end_hour=end))
                elif d in secondary_days:
                    start, end = self.secondary_weekday_away.sample()
                    interval.append(TimeRange(start_hour=start, end_hour=end))
                else:
                    interval.append(None)
            return interval
        

class SingleOccupantTracker:
    def __init__(self, role: OccupantRole, assumptions: OccupancyAssumptions):
        self.role = role
        self.assumptions = assumptions

    def sample_weekly(self, household_away_intervals: list[TimeRange | None]) -> list[TimeRange | None]:
        """Samples the weekly leave and return times for this occupant based on their role and household patterns"""
        
            
                