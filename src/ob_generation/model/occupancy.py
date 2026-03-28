from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, computed_field, model_validator


class ScheduleRigidness(str, Enum):
    """How consistently a time pattern is followed.
    Maps internally to a standard deviation applied to start/end hours.
    - HIGHLY_VARIABLE: schedule shifts by ~2 hrs regularly
    - SOMEWHAT_VARIABLE: schedule shifts by ~1 hr regularly
    - MOSTLY_CONSISTENT: schedule shifts by ~30 min regularly
    - STRICT: schedule shifts by ~15 min regularly
    """
    HIGHLY_VARIABLE = "highly_variable"
    SOMEWHAT_VARIABLE = "somewhat_variable"
    MOSTLY_CONSISTENT = "mostly_consistent"
    STRICT = "strict"

    def to_std_dev_hours(self) -> float:
        """Convert rigidness level to a standard deviation in hours."""
        mapping = {
            ScheduleRigidness.HIGHLY_VARIABLE: 2.0,
            ScheduleRigidness.SOMEWHAT_VARIABLE: 1.0,
            ScheduleRigidness.MOSTLY_CONSISTENT: 0.5,
            ScheduleRigidness.STRICT: 0.25
        }
        return mapping[self]


class TimeRange(BaseModel):
    """
    Represents a time range with start and end hours.
    Attributes:
        start_hour (float): The hour of the day [0-24) when the time range starts.
        end_hour (float): The hour of the day [0-24) when the time range ends.
    """
    start_hour: float = Field(default=0, ge=0, lt=24)
    end_hour: float = Field(default=0, ge=0, lt=24)

    @computed_field
    @property
    def wraps_midnight(self) -> bool:
        """True if range crosses midnight."""
        return self.end_hour < self.start_hour

    def contains_hour(self, hour: float) -> bool:
        """Check if a given hour falls within this time range [start_hour, end_hour)."""
        if not self.wraps_midnight:
            return self.start_hour <= hour < self.end_hour
        else:
            return hour >= self.start_hour or hour < self.end_hour
        
    def overlaps(self, other: 'TimeRange') -> bool:
        """Check if this time range overlaps with another time range."""
        # Check if either start or end hour of one range is contained in the other
        return (self.contains_hour(other.start_hour) or self.contains_hour(other.end_hour) or
                other.contains_hour(self.start_hour) or other.contains_hour(self.end_hour))


class MobilityCluster(str, Enum):
    """Mobility cluster for a single occupant.
    Maps to a Markov-chain centroid (cluster index 0–4) in the generator.
    - MOSTLY_HOME: near-zero away probability all day
    - LONG_DAY_AWAY: away from early morning through evening (work, school, day trip)
    - MORNING_AWAY: away peaking early morning through late morning (~4am–12pm)
    - AFTERNOON_AWAY: away peaking from late morning through early evening
    - EVENING_NIGHT_AWAY: away rising into evening and night
    """
    MOSTLY_HOME = "mostly_home"
    LONG_DAY_AWAY = "long_day_away"
    MORNING_AWAY = "morning_away"
    AFTERNOON_AWAY = "afternoon_away"
    EVENING_NIGHT_AWAY = "evening_night_away"

    def to_cluster_index(self) -> int:
        """Convert mobility cluster to a corresponding index."""
        mapping = {
            MobilityCluster.MOSTLY_HOME: 0,
            MobilityCluster.LONG_DAY_AWAY: 1,
            MobilityCluster.MORNING_AWAY: 2,
            MobilityCluster.AFTERNOON_AWAY: 3,
            MobilityCluster.EVENING_NIGHT_AWAY: 4
        }
        return mapping[self]


class OccupantMobilityProfile(BaseModel):
    """Weekday and weekend mobility cluster assignment for a single occupant."""
    weekday_cluster: MobilityCluster
    weekend_cluster: MobilityCluster


class HouseholdComposition(BaseModel):
    """Per-occupant mobility cluster assignments for the household."""
    occupants: list[OccupantMobilityProfile]


class WeekdayOccupancyPattern(BaseModel):
    """Defines the weekday unoccupied patterns."""
    is_always_occupied: bool = Field(...)
    away_interval: Optional[TimeRange] = Field(None)
    num_of_days: Optional[int] = Field(None, ge=1, le=5)
    away_time_rigidness: Optional[ScheduleRigidness] = Field(None)

class WeekendOccupancyPattern(BaseModel):
    """Defines the weekend unoccupied patterns."""
    is_always_occupied: bool = Field(...)
    away_interval: Optional[TimeRange] = Field(None)
    away_time_rigidness: Optional[ScheduleRigidness] = Field(None)

class SleepPattern(BaseModel):
    """Defines the typical sleep time pattern for the household."""
    is_always_awake: bool = Field(...)
    sleep_time: Optional[TimeRange] = Field(None)
    sleep_time_rigidness: Optional[ScheduleRigidness] = Field(None)

class Occupancy(BaseModel):
    """The occupancy of the household (number of occupants, household composition, occupancy patterns)."""
    num_occupants: int = Field(..., ge=1)
    household_composition: Optional[HouseholdComposition] = None
    weekday_pattern: Optional[WeekdayOccupancyPattern] = None
    weekend_pattern: Optional[WeekendOccupancyPattern] = None
    sleep_pattern: Optional[SleepPattern] = None

    @model_validator(mode='after')
    def validate_household_composition(self):
        """Validate that the number of occupant profiles matches num_occupants."""
        if self.household_composition is None:
            return self
        n = len(self.household_composition.occupants)
        if n != self.num_occupants:
            raise ValueError(f"household_composition has {n} occupant(s), expected {self.num_occupants}.")
        return self
