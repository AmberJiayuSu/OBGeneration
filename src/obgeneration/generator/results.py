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
    laundry_dhw_event_schedule: list[list[int]] | None = None
    dishwasher_dhw_event_schedule: list[list[int]] | None = None
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


class WindowResult(BaseModel):
    """Opening-area fraction schedule for a household's windows.

    The schedule is the fraction itself (0–1), the quantity
    `ZoneVentilation:WindandStackOpenArea` reads as Opening Area Fraction.
    """

    annual_schedule: list[float]
    summer_design_day_schedule: list[float] | None = None
    winter_design_day_schedule: list[float] | None = None
    peak_units: Literal["fraction"] = "fraction"


class OccupantBehaviorResult(BaseModel):
    """One household's schedules, in the shape the stochastic IDF path consumes.

    `hvac.heating` and `hvac.cooling` map to the separate heating and cooling
    setpoint schedules. `dhw` maps to water use. `window` is reserved for an
    opening-fraction schedule and is `None` until that schedule is specified.
    Occupancy states and appliance event schedules stay on `occupancy` and
    `equipment` for the steps that need them; they are not themselves IDF
    schedules.
    """

    occupancy: OccupancyResult
    lighting: LightingResult
    equipment: EquipmentResult
    dhw: DHWResult
    hvac: HVACResult
    window: WindowResult | None = None
