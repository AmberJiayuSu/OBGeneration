from typing import Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum



class TurnOffHabits(BaseModel):
    when_house_empty: bool = Field(default=True)
    when_room_empty: bool = Field(default=True)
    when_daylight_bright: bool = Field(default=False)

class Lighting(BaseModel):
    """
    Represents the lighting profile of a household.
    Attributes:
       
    """
    turn_off_habits: TurnOffHabits = Field(default_factory=TurnOffHabits)

