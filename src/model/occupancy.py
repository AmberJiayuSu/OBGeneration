from typing import Optional
from pydantic import BaseModel, Field, computed_field

class HouseholdComposition(BaseModel):
    """
    Represents the composition of a household based on daily mobility patterns.
    Attributes:
        daily_commuter (int): Number of adults who COMMUTE to a workplace 5 days a week.
        hybrid_worker (int): Number of hybrid workers (1-4 days commute).
        stayathome (int): Number of people home most of the day (remote, retired, stay home parent etc.).
        daily_school_or_daycare (int): Number of children attending school/daycare daily.
        college_student (int): Number of college students with rather high variance in daily schedules.
    """
    daily_commuter: int = Field(default=0, ge=0)
    hybrid_worker: int = Field(default=0, ge=0)
    stayathome: int = Field(default=0, ge=0)
    k12_or_daycare: int = Field(default=0, ge=0)
    college_student: int = Field(default=0, ge=0)

class TimeRange(BaseModel):
    """
    Represents a time range with start and end hours.
    Attributes:
        start_hour (int): The hour of the day (0-23) when the time range starts.
        end_hour (int): The hour of the day (0-23) when the time range ends.
    """
    start_hour: int = Field(default=0, ge=0, le=23) 
    end_hour: int = Field(default=0, ge=0, le=23)

    @computed_field
    @property
    def wraps_midnight(self) -> bool:
        """True if range crosses midnight."""
        return self.end_hour < self.start_hour

    def contains_hour(self, hour: int) -> bool:
        """Check if a given hour falls within this time range (exclusive of boundaries)."""
        if not self.wraps_midnight:
            return self.start_hour < hour < self.end_hour
        else:
            return hour > self.start_hour or hour < self.end_hour

class WeekdayAwayInterval(BaseModel):
    """Details for a specific weekday away interval."""
    num_of_days: int = Field(..., ge=1, le=5)
    away_interval: TimeRange = Field(...)

class WeekdayPattern(BaseModel):
    """Defines the unoccupied patterns for weekdays."""
    is_always_occupied: bool = Field(...)
    primary_away_interval: Optional[WeekdayAwayInterval] = Field(None)
    secondary_away_interval: Optional[WeekdayAwayInterval] = Field(None)


class WeekendPattern(BaseModel):
    """Defines the unoccupied patterns for weekends."""
    is_always_occupied: bool = Field(...)
    away_interval: Optional[TimeRange] = Field(None)


class Occupancy(BaseModel):
    """The occupancy of the household (number of occupants, household composition, occupancy patterns)."""
    num_occupants: int = Field(..., ge=1)
    household_composition: HouseholdComposition
    weekday_pattern: WeekdayPattern
    weekend_pattern: WeekendPattern
    sleep_time: TimeRange

