from typing import Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class Lighting(BaseModel):
    """
    Represents the lighting profile of a household.
    Attributes:
       
    """
    when_away: bool = Field(..., description="Indicates if lighting is on when people are away.")
    when_daylight_bright: bool = Field(..., description="Indicates if lighting is adjusted based on daylight brightness.")

