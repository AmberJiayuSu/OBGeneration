from typing import Optional
from pydantic import BaseModel, Field, computed_field


class HouseholdComposition(BaseModel):
    """
    Represents the composition of a household based on daily mobility patterns.
    Attributes:
        daily_commuter (int): Number of adults who COMMUTE to a workplace 5 days a week.
        hybrid_worker (int): Number of hybrid workers (1-4 days commute).
        stayathome (int): Number of people home most of the day (remote, retired, etc.).
        daily_school_or_daycare (int): Number of children attending school/daycare daily.
        college_student (int): Number of college students.
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
        """Automatically computed: True if range crosses midnight."""
        return self.end_hour < self.start_hour

    def contains_hour(self, hour: int) -> bool:
        """Check if a given hour falls within this time range (exclusive of boundaries)."""
        if not self.wraps_midnight:
            return self.start_hour < hour < self.end_hour
        else:
            return hour > self.start_hour or hour < self.end_hour

class OccupancyPattern(BaseModel):
    """
    Base pattern for occupancy (used for Weekends directly).
    Attributes:
        is_always_occupied (bool): If True, assume someone is always home.
        no_one_home_interval (Optional[TimeRange]): Time range when house is empty (if not always occupied).
        sleep_time (Optional[TimeRange]): Typical sleep hours.
    """
    is_always_occupied: bool = Field(default=False)
    no_one_home_interval: Optional[TimeRange] = None
    sleep_time: Optional[TimeRange] = None

class WeekdayOccupancyPattern(OccupancyPattern):
    """
    Weekday specific pattern including frequency of vacancy.
    Attributes:
        vacancy_frequency (Optional[int]): How many weekdays per week the house is empty.
    """
    vacancy_frequency: Optional[int] = Field(default=None, ge=1, le=5)

class Occupancy(BaseModel):
    """
    Represents the occupancy profile of a household.
    """
    num_occupants: int = Field(ge=1)
    household_composition: HouseholdComposition
    weekday_occupancy_pattern: WeekdayOccupancyPattern
    weekend_occupancy_pattern: OccupancyPattern