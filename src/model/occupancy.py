from typing import Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class HouseholdComposition(BaseModel):
    """
    Represents the composition of a household.
    Attributes:
        fulltime_workers (int): Number of full-time workers in the household.
        hybrid_workers (int): Number of hybrid workers in the household.
        stayathome (int): Number of stay-at-home individuals in the household.
        k12 (int): Number of K-12 students in the household.
        college_students (int): Number of college students in the household.
    """
    fulltime_workers: int = Field(default=0,ge=0)
    hybrid_workers: int = Field(default=0,ge=0)
    stayathome: int = Field(default=0,ge=0)
    k12: int = Field(default=0,ge=0)
    college_students: int = Field(default=0,ge=0)

    
    
class TimeRange(BaseModel):
    """
    Represents a time range with start and end hours.
    Attributes:
        start_hour (int): The hour of the day (0-23) when the time range starts, exclusive.
        end_hour (int): The hour of the day (0-23) when the time range ends, exclusive.
        wraps_midnight (bool): True if the range crosses midnight (end_hour < start_hour logically means next day)
    """
    start_hour: int = Field(default=0, ge=0, le=23) 
    end_hour: int = Field(default=0, ge=0, le=23)

    @computed_field
    @property
    def wraps_midnight(self) -> bool:
        """Automatically computed: True if range crosses midnight."""
        return self.end_hour < self.start_hour

    
    def contains_hour(self, hour: int) -> bool:
        """Check if a given hour falls within this time range."""
        if not self.wraps_midnight:
            return self.start_hour < hour < self.end_hour
        else:
            return hour > self.start_hour or hour < self.end_hour
    

class Occupancy(BaseModel):
    """
    Represents the occupancy profile of a household.
    """
    num_occupants: int = Field(ge=1)
    household_composition: HouseholdComposition
    weekday_no_one_home: Optional[TimeRange] = None
    weekend_no_one_home: Optional[TimeRange] = None
    weekday_sleep_time: Optional[TimeRange] = None
    weekend_sleep_time: Optional[TimeRange] = None

