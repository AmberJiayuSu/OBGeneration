from pydantic import BaseModel, Field
from enum import Enum

class OccupancyState(Enum):
    """
    Enumeration of possible occupancy states.
    """
    HOME = "home"
    AWAY = "away"
    SLEEPING = "sleeping"

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
    fulltime_workers: int = Field(ge=0)
    hybrid_workers: int = Field(ge=0)
    stayathome: int = Field(ge=0)
    k12: int = Field(ge=0)
    college_students: int = Field(ge=0)

class TimeRange(BaseModel):
    """
    Represents a time range with start and end hours.
    Attributes:
        start_hour (int): The hour of the day (0-23) when the time range starts.
        end_hour (int): The hour of the day (0-23) when the time range ends.
    """
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=23)

class Occupancy(BaseModel):
    """
    Represents the occupancy profile of a household.
    """
    num_occupants: int = Field(ge=1)
    household_composition: HouseholdComposition
    weekday_no_one_home: TimeRange
    weekend_no_one_home: TimeRange
    weekday_sleep_time: TimeRange
    weekend_sleep_time: TimeRange

enu

class OccupancyWeeklyScheduleGeneration:
    """
    Class for generating weekly occupancy schedules.
    """
    def __init__(self, occupancy: Occupancy):
        self.occupancy = occupancy

    def generate_schedule(self):
        raise NotImplementedError("This method should be implemented to generate the weekly occupancy schedule.")
    
    def generate_weekday_schedule(self):
        raise NotImplementedError("This method should be implemented to generate the weekday occupancy schedule.")
    
    def generate_weekend_schedule(self):
        raise NotImplementedError("This method should be implemented to generate the weekend occupancy schedule.")