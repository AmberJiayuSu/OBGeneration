from typing import Optional
from pydantic import BaseModel, Field, computed_field, model_validator

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


class WeekdayOccupancyPattern(BaseModel):
    """Defines the weekday unoccupied patterns."""
    is_always_occupied: bool = Field(...)
    away_interval: Optional[TimeRange] = Field(None)
    num_of_days: Optional[int] = Field(None, ge=1, le=5)

class WeekendOccupancyPattern(BaseModel):
    """Defines the weekend unoccupied patterns."""
    is_always_occupied: bool = Field(...)
    away_interval: Optional[TimeRange] = Field(None)

class Occupancy(BaseModel):
    """The occupancy of the household (number of occupants, household composition, occupancy patterns)."""
    num_occupants: int = Field(..., ge=1)
    household_composition: HouseholdComposition
    weekday_pattern: WeekdayOccupancyPattern
    weekend_pattern: WeekendOccupancyPattern
    sleep_time: TimeRange

    @model_validator(mode='after')
    def validate_household_composition(self):
        """Validate that the sum of all roles equals num_occupants."""
        composition = self.household_composition
        total = (
            composition.daily_commuter +
            composition.hybrid_worker +
            composition.stayathome +
            composition.k12_or_daycare +
            composition.college_student
        )
        if total != self.num_occupants:
            raise ValueError(
                f"Sum of household composition ({total}) does not match num_occupants ({self.num_occupants}). "
                f"Breakdown: daily_commuter={composition.daily_commuter}, "
                f"hybrid_worker={composition.hybrid_worker}, "
                f"stayathome={composition.stayathome}, "
                f"k12_or_daycare={composition.k12_or_daycare}, "
                f"college_student={composition.college_student}"
            )
        return self

