from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


from enum import Enum


class WindowOpeningBehavior(str, Enum):
    """
    Low-dimensional household window-opening behavior pattern.

    The categories are intentionally behavioral rather than purely frequency-based.
    The key distinction is whether window opening is coordinated with mechanical
    HVAC operation or may occur while HVAC remains on.
    """

    ALWAYS_CLOSED = "always_closed"
    """
    Windows are never opened.
    """

    NATURAL_VENTILATION = "natural_ventilation"
    """
    Windows are used intentionally as a substitute for mechanical HVAC
    when outdoor conditions are favorable.
    """

    OCCASIONALLY_OPEN = "occasionally_open"
    """
    Windows are opened occasionally for short.
    """

    LONG_OPEN = "long_open"
    """
    Windows are left open for extended periods.
    """

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