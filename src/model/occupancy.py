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

class AwayPattern(BaseModel):
    """Details for a specific weekday away interval."""
    num_of_days: int = Field(..., ge=1, le=5)
    away_interval: TimeRange = Field(...)

class OcccupancyPattern(BaseModel):
    """Defines the unoccupied patterns."""
    is_always_occupied: bool = Field(...)
    away_pattern: Optional[AwayPattern] = Field(None)


class Occupancy(BaseModel):
    """The occupancy of the household (number of occupants, household composition, occupancy patterns)."""
    num_occupants: int = Field(..., ge=1)
    household_composition: HouseholdComposition
    weekday_pattern: OcccupancyPattern
    weekend_pattern: OcccupancyPattern
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

