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


class ClusterComposition(BaseModel):
    """Number of household members assigned to each mobility cluster.
    All counts must sum to num_occupants. The same 5 clusters apply to both
    weekday and weekend; the household can have different compositions for each.
    - mostly_home: near-zero away probability all day
    - long_day_away: away from early morning through evening (work, school, day trip)
    - morning_away: away peaking in the early morning through late morning (~4am–12pm)
    - afternoon_away: away peaking from late morning through early evening
    - evening_night_away: away rising into evening and night
    """
    mostly_home: int = Field(default=0, ge=0)
    long_day_away: int = Field(default=0, ge=0)
    morning_away: int = Field(default=0, ge=0)
    afternoon_away: int = Field(default=0, ge=0)
    evening_night_away: int = Field(default=0, ge=0)


class HouseholdComposition(BaseModel):
    """Mobility cluster composition of the household for weekdays and weekends."""
    weekday: ClusterComposition
    weekend: ClusterComposition


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
        """Validate that cluster counts sum to num_occupants for both weekday and weekend."""
        if self.household_composition is None:
            return self
        wd = self.household_composition.weekday
        we = self.household_composition.weekend
        wd_total = wd.mostly_home + wd.long_day_away + wd.morning_away + wd.afternoon_away + wd.evening_night_away
        we_total = we.mostly_home + we.long_day_away + we.morning_away + we.afternoon_away + we.evening_night_away
        if wd_total != self.num_occupants:
            raise ValueError(f"Weekday cluster counts sum to {wd_total}, expected {self.num_occupants}.")
        if we_total != self.num_occupants:
            raise ValueError(f"Weekend cluster counts sum to {we_total}, expected {self.num_occupants}.")
        return self
