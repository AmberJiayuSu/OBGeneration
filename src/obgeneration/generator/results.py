from __future__ import annotations

from typing import Literal
from typing import TYPE_CHECKING

from collections.abc import Sequence

from pydantic import BaseModel,  Field

if TYPE_CHECKING:
    from obgeneration.generator.occupancy_generator import HouseholdOccupancyFractions


class FractionalScheduleResult(BaseModel):
    """Base result for schedules represented as a peak plus normalized fractions."""

    peak_value: float = Field(..., ge=0)
    schedule: Sequence[float]


class TemperatureScheduleResult(BaseModel):
    """Result for schedules represented as absolute temperatures."""

    schedule: Sequence[float]


class OccupancyResult(FractionalScheduleResult):
    """Occupancy output plus raw occupancy-state detail."""

    occupancy_states: Sequence[Sequence["HouseholdOccupancyFractions"]]
    peak_units:  Literal["people"] = "people"


class LightingResult(FractionalScheduleResult):
    """Lighting output plus dimming metadata."""

    dimming_enabled: bool
    peak_units: Literal["fraction"] = "fraction"


class EquipmentResult(FractionalScheduleResult):
    """Equipment output plus appliance-cycle metadata used downstream."""

    laundry_cycles: Sequence[Sequence[int]]
    dishwasher_cycles: Sequence[Sequence[int]]
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

