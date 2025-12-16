from typing import Literal, Union, Annotated, Optional
from pydantic import BaseModel, Field




# No control
class NoControl(BaseModel):
    type: Literal["no_control"] = "no_control"

# Binary control
class BinaryControl(BaseModel):
    type: Literal["binary"] = "binary"
    active: bool = Field(default=True)
    sleep: bool = Field(default=False)
    absent: bool = Field(default=False)

# Valve control for heating systems
class ValveControl(BaseModel):
    type: Literal["valve"] = "valve"
    active_percentage: Optional[float] = Field(None, description="Active Valve Percentage")
    sleep_percentage: Optional[float] = Field(None, description="Sleep Valve Percentage")
    absent_percentage: Optional[float] = Field(None, description="Absent Valve Percentage")
    

# Thermostat
class ThermostatControl(BaseModel):
    type: Literal["thermostat"] = "thermostat"
    active_setpoint: Optional[float] = Field(None, description="Active Temp C")
    sleep_setpoint: Optional[float] = Field(None, description="Sleep Temp C")
    absent_setpoint: Optional[float] = Field(None, description="Absent Temp C")


HeatingSystem = Annotated[
    Union[NoControl, BinaryControl, ValveControl, ThermostatControl], 
    Field(discriminator="type")
]

CoolingSystem = Annotated[
    Union[NoControl, BinaryControl, ThermostatControl], 
    Field(discriminator="type")
]


class HVAC(BaseModel):
    """
    heating: Can be a specific control system OR None (No heating).
    cooling: Can be a specific control system OR None (No cooling).
    """
    
    heating: Optional[HeatingSystem] = Field(
        default_factory=lambda: ThermostatControl(
            active_setpoint=20.0, 
            sleep_setpoint=18.0, 
            absent_setpoint=15.0
        )
    )

    cooling: Optional[CoolingSystem] = Field(default=None)


