from typing import Literal, Union, Annotated, Optional
from pydantic import BaseModel, Field
from enum import Enum


class IntensityLevel(str, Enum):
    """Intensity levels for valve control (minimum=frost prevention*, low=~25%, medium=~50%, high=~75%, fullon=100%)"""
    MINIMUM = "minimum"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FULLON = "fullon"




# No control
class NoControl(BaseModel):
    type: Literal["no_control"] = "no_control"

# Binary Control
class BinaryControl(BaseModel):
    type: Literal["binary"] = "binary"
    active_state: bool = Field(description="Intensity level when occupants are active")
    sleep_state: bool = Field(description="Intensity level during sleep")
    absent_state: bool = Field(description="Intensity level when absent")


# Valve control for heating systems
class ValveControl(BaseModel):
    type: Literal["valve"] = "valve"
    active_level: IntensityLevel = Field(description="Intensity level when occupants are active")
    sleep_level: IntensityLevel = Field(description="Intensity level during sleep")
    absent_level: IntensityLevel = Field(description="Intensity level when absent")
    

# Thermostat
class ThermostatControl(BaseModel):
    type: Literal["thermostat"] = "thermostat"
    active_setpoint: Optional[float] = Field(None, description="Active Temp C")
    sleep_setpoint: Optional[float] = Field(None, description="Sleep Temp C")
    absent_setpoint: Optional[float] = Field(None, description="Absent Temp C")


HeatingSystem = Annotated[
    Union[NoControl, BinaryControl,ValveControl, ThermostatControl], 
    Field(discriminator="type")
]

CoolingSystem = Annotated[
    Union[NoControl, BinaryControl,ThermostatControl], 
    Field(discriminator="type")
]


class HVAC(BaseModel):
    """
    heating: Can be a specific control system OR None (No heating).
    cooling: Can be a specific control system OR None (No cooling).
    """
    
    heating: Optional[HeatingSystem] = Field(
        default_factory=lambda: ThermostatControl(
            active_setpoint=22.0, 
            sleep_setpoint=21.0, 
            absent_setpoint=20.0
        )
    )

    cooling: Optional[CoolingSystem] = Field(default=None)


