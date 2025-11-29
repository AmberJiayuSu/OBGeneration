from typing import Optional, Annotated, Union, Literal
from pydantic import BaseModel, Field

class Setpoint(BaseModel):
    setpoint_celsius: Optional[float] = Field(default=None)
    setback_sleep_celsius: Optional[float] = Field(default=None)
    setback_absent_celsius: Optional[float] = Field(default=None)


class HeatingValve(BaseModel):
    heating_valve_active_percentage: Optional[float] = Field(default=None)
    heating_valve_sleep_percentage: Optional[float] = Field(default=None)
    heating_valve_absent_percentage: Optional[float] = Field(default=None)

class OnOffControl(BaseModel):
    active: bool = Field(default=True)
    sleep: bool = Field(default=False)
    absent: bool = Field(default=False)


# Discriminated union types for the temperature_control subfield (matches schema oneOf)
class ThermostatControl(BaseModel):
    type: Literal["thermostat_setpoint"]
    setpoint: Setpoint


class ValveControl(BaseModel):
    type: Literal["valve"]
    valve: HeatingValve


class NoControl(BaseModel):
    type: Literal["none"]
    onoff: OnOffControl


HeatingControl = Annotated[Union[ThermostatControl, ValveControl, NoControl], Field(discriminator="type")]
CoolingControl = Annotated[Union[ThermostatControl, NoControl], Field(discriminator="type")]

class Heating(BaseModel):
    """Represents the heating profile of a household.

    Notes:
    - Field names use `onoff_control` and `temperature_control` to match the JSON schema.
    - `temperature_control` is a discriminated union with `type` as the discriminator.
    """
    onoff_control: bool = Field(default=True)
    control: HeatingControl = Field(default_factory=lambda: ThermostatControl(type="thermostat_setpoint", setpoint=Setpoint()))




class Cooling(BaseModel):
    onoff_control: bool = Field(default=True)
    control: CoolingControl = Field(default_factory=lambda: ThermostatControl(type="thermostat_setpoint", setpoint=Setpoint()))


class HVAC(BaseModel):
    heating: Heating = Field(default_factory=Heating)
    cooling: Cooling = Field(default_factory=Cooling)




