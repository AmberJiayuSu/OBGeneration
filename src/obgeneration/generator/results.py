from __future__ import annotations

from typing import Literal

from pydantic import BaseModel,  Field

from obgeneration.generator.types import HouseholdOccupancyFractions


class FractionalScheduleResult(BaseModel):
    """Base result for schedules represented as a peak plus normalized fractions."""
    peak_value: float = Field(..., ge=0)
    annual_schedule: list[float]
    summer_design_day_schedule: list[float] | None = None
    winter_design_day_schedule: list[float] | None = None


class TemperatureScheduleResult(BaseModel):
    """Result for schedules represented as absolute temperatures."""
    annual_schedule: list[float]
    summer_design_day_schedule: list[float] | None = None
    winter_design_day_schedule: list[float] | None = None


class OccupancyResult(FractionalScheduleResult):
    """Occupancy output plus raw occupancy-state detail."""
    occupancy_states: list[list[HouseholdOccupancyFractions]]
    peak_units:  Literal["people"] = "people"


class LightingResult(FractionalScheduleResult):
    """Lighting output plus dimming metadata."""
    dimming_enabled: bool
    peak_units: Literal["W/m2"] = "W/m2"


class EquipmentResult(FractionalScheduleResult):
    """Equipment output plus appliance-cycle metadata used downstream."""
    laundry_cycles: list[list[int]]
    dishwasher_cycles: list[list[int]]
    peak_units: Literal["W"] = "W"

class DHWResult(FractionalScheduleResult):
    """DHW output represented as normalized flow fractions."""
    peak_units: Literal["m3/s"] = "m3/s"


class SetpointResult(TemperatureScheduleResult):
    """Heating/cooling output represented as absolute temperatures."""
    peak_units: Literal["°C"] = "°C"


class HVACResult(BaseModel):
    """Combined HVAC output with optional heating and cooling schedules."""
    heating: SetpointResult | None = None
    cooling: SetpointResult | None = None
