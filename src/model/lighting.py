from typing import Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum

class LightingBehavior(Enum):
    ALWAYS_ON = "always_on"
    MOSTLY_ON = "mostly_on"
    NECESSARY_ON = "necessary_on"
    SENSOR = "sensor"

class Lighting(BaseModel):
    """
    Represents the lighting profile of a household.
    Attributes:
       
    """
    LED: bool = Field(default=True)
    usage_pattern: LightingBehavior = Field(default=LightingBehavior.NECESSARY_ON)

