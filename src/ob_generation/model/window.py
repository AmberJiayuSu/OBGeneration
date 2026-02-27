from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class WindowOpeningBehavior(str, Enum):
    """Window opening behavior patterns."""
    FREQUENTLY_OPEN = "frequently_open"
    OCCASIONALLY_OPEN = "occasionally_open"
    ALWAYS_CLOSED = "always_closed"


class Window(BaseModel):
    """Represents the window operation characteristics of a household."""
    heating_season: Optional[WindowOpeningBehavior] = Field(
        default=None,
        description="Window opening behavior during heating season"
    )
    cooling_season: Optional[WindowOpeningBehavior] = Field(
        default=None,
        description="Window opening behavior during cooling season"
    )
    shoulder_season: Optional[WindowOpeningBehavior] = Field(
        default=None,
        description="Window opening behavior during shoulder seasons"
    )