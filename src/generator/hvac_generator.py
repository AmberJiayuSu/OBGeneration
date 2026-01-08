
from generator.occupancy_generator import OccupancyGenerator
import model.hvac as HVAC
from pydantic import BaseModel, Field
import json
from pathlib import Path


class TRVAssumptions(BaseModel):
    """Defines the temperature mapping for TRV (Thermostatic Radiator Valve) intensity levels."""
    valve_minimum: float = Field(..., description="Minimum valve (frost prevention) temperature in Celsius")
    valve_low: float = Field(..., description="Low valve (~25%) temperature in Celsius")
    valve_medium: float = Field(..., description="Medium valve (~50%) temperature in Celsius")
    valve_high: float = Field(..., description="High valve (~75%) temperature in Celsius")
    valve_fullon: float = Field(..., description="Fullon valve (100%) temperature in Celsius")

class HeatingDefaultSetpoints(BaseModel):
    """Default setpoints for heating thermostat control."""
    active_setpoint: float = Field(22.0, description="Active heating setpoint in Celsius")
    sleep_setpoint: float = Field(21.0, description="Sleep heating setpoint in Celsius")
    absent_setpoint: float = Field(20.0, description="Absent heating setpoint in Celsius")

class CoolingDefaultSetpoints(BaseModel):
    """Default setpoints for heating thermostat control."""
    active_setpoint: float = Field(23.0, description="Active heating setpoint in Celsius")
    sleep_setpoint: float = Field(22.0, description="Sleep heating setpoint in Celsius")
    absent_setpoint: float = Field(24.0, description="Absent heating setpoint in Celsius")


class HVACAssumptions(BaseModel):
    """Master configuration for HVAC behavioral assumptions."""
    trv: TRVAssumptions
    minimum_heating_setpoint: float = Field(7.0, description="Minimum heating setpoint in Celsius")
    maximum_cooling_setpoint: float = Field(35.0, description="Maximum cooling setpoint in Celsius")
    heating_defaults: HeatingDefaultSetpoints = Field(
        ...,description="Default setpoints for heating thermostat control"
    )
    cooling_defaults: CoolingDefaultSetpoints = Field(
        ...,description="Default setpoints for cooling thermostat control"
    )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "HVACAssumptions":
        """Load HVAC assumptions from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def default(cls) -> "HVACAssumptions":
        """Returns the standard/default assumptions for HVAC systems."""
        return cls(
            trv=TRVAssumptions(
                valve_minimum=7.0,
                valve_low=14.0,
                valve_medium=18.0,
                valve_high=22.0,
                valve_fullon=26.0,
            ),
            minimum_heating_setpoint=7.0,
            maximum_cooling_setpoint=35.0,
            heating_defaults=HeatingDefaultSetpoints(
                active_setpoint=22.0,
                sleep_setpoint=21.0,
                absent_setpoint=20.0
            ),
            cooling_defaults=CoolingDefaultSetpoints(
                active_setpoint=23.0,
                sleep_setpoint=22.0,
                absent_setpoint=24.0
            )
        )


class HVACGenerator:
    def __init__(self, hvac: HVAC.HVAC, assumptions: HVACAssumptions):
        self.hvac = hvac
        self.assumptions = assumptions

    def _get_trv_temp(self, level: HVAC.IntensityLevel) -> float:
        """Helper method to map intensity level to temperature."""
        trv = self.assumptions.trv
        level_to_temp = {
            HVAC.IntensityLevel.MINIMUM: trv.valve_minimum,
            HVAC.IntensityLevel.LOW: trv.valve_low,
            HVAC.IntensityLevel.MEDIUM: trv.valve_medium,
            HVAC.IntensityLevel.HIGH: trv.valve_high,
            HVAC.IntensityLevel.FULLON: trv.valve_fullon,
        }
        if level not in level_to_temp:
            raise ValueError(f"Unknown intensity level: {level}")
        return level_to_temp[level]

    def heating_setpoint_annual_schedule(self, occupancy_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        if self.hvac.heating is None:
            return None
        
        assert len(occupancy_mask_annual) == len(sleep_mask_annual), "occupancy and sleep must have same #weeks"
        assert len(occupancy_mask_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_occupancy, weekly_sleep_mask in zip(occupancy_mask_annual, sleep_mask_annual):
            assert len(weekly_occupancy) == len(weekly_sleep_mask), "weekly occupancy/sleep length mismatch"
            schedule.append(self.heating_setpoint_weekly_schedule(weekly_occupancy, weekly_sleep_mask))
        return schedule
        
       
    def heating_setpoint_weekly_schedule(self, occupancy_mask_weekly: list[bool], sleep_mask_weekly: list[bool]) -> list[float]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        heating = self.hvac.heating
        minimum_setpoint = self.assumptions.minimum_heating_setpoint
        #if no control at all.
        if heating.type=="no_control":
            schedule = [self.assumptions.heating_defaults.active_setpoint for _ in range(len (occupancy_mask_weekly))] # default 22C always on
            return schedule
        #if binary control
        elif heating.type=="binary":
            # start with active state
            base_temp = self.assumptions.heating_defaults.active_setpoint if heating.active_state else minimum_setpoint
            schedule = [base_temp for _ in range(len(occupancy_mask_weekly))]

            # adjust for sleep
            sleep_temp = self.assumptions.heating_defaults.sleep_setpoint if heating.sleep_state else minimum_setpoint
            schedule = OccupancyGenerator.revise_by_sleep(sleep_mask_weekly, schedule, sleep_temp)

            # adjust for absence
            absent_temp = self.assumptions.heating_defaults.absent_setpoint if heating.absent_state else minimum_setpoint
            schedule = OccupancyGenerator.revise_by_absence(occupancy_mask_weekly, schedule, absent_temp)
            return schedule
        # if valve control
        elif heating.type=="valve":
            # start with active level
            base_temp = self._get_trv_temp(heating.active_level)
            schedule = [base_temp for _ in range(len(occupancy_mask_weekly))]

            # adjust for sleep
            sleep_temp = self._get_trv_temp(heating.sleep_level)
            schedule = OccupancyGenerator.revise_by_sleep(sleep_mask_weekly, schedule, sleep_temp)

            # adjust for absence
            absent_temp = self._get_trv_temp(heating.absent_level)
            schedule = OccupancyGenerator.revise_by_absence(occupancy_mask_weekly, schedule, absent_temp)
            return schedule
       
        # if setpoint control
        elif heating.type=="thermostat":
            if heating.active_setpoint is not None:
                schedule =  [heating.active_setpoint for _ in range(len (occupancy_mask_weekly))]
            else:
                schedule = [minimum_setpoint for _ in range(len (occupancy_mask_weekly))]
            # adjust for setback
            if heating.sleep_setpoint is not None:
                setback = heating.sleep_setpoint
                schedule = OccupancyGenerator.revise_by_sleep(sleep_mask_weekly,schedule, setback)
            else:
                schedule = OccupancyGenerator.revise_by_sleep(sleep_mask_weekly, schedule, minimum_setpoint)
            # adjust for absence
            if heating.absent_setpoint is not None:
                setback = heating.absent_setpoint
                schedule = OccupancyGenerator.revise_by_absence(occupancy_mask_weekly, schedule, setback)
            else:
                schedule = OccupancyGenerator.revise_by_absence(occupancy_mask_weekly, schedule, minimum_setpoint)
            return schedule
        else:
            raise NotImplementedError(f"Heating type {heating.type} not yet implemented.")



    def cooling_setpoint__annual_schedule(self, occupancy_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        if self.hvac.cooling is None:
            return None
        
        assert len(occupancy_mask_annual) == len(sleep_mask_annual), "occupancy and sleep must have same #weeks"
        assert len(occupancy_mask_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_occupancy, weekly_sleep_mask in zip(occupancy_mask_annual, sleep_mask_annual):
            assert len(weekly_occupancy) == len(weekly_sleep_mask), "weekly occupancy/sleep length mismatch"
            schedule.append(self.cooling_setpoint_weekly_schedule(weekly_occupancy, weekly_sleep_mask))
        return schedule


    def cooling_setpoint_weekly_schedule(self, occupied_time_mask_weekly: list[bool], sleep_time_mask_weekly: list[bool]) -> list[float]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        cooling = self.hvac.cooling
        cooling_max_setpoint = self.assumptions.maximum_cooling_setpoint
        #if no control at all.
        if cooling.type=="no_control":
            schedule = [self.assumptions.cooling_defaults.active_setpoint for _ in range(len (occupied_time_mask_weekly))]
            return schedule
        #if binary control
        elif cooling.type=="binary":
            # start with active state
            base_temp = self.assumptions.cooling_defaults.active_setpoint if cooling.active_state else cooling_max_setpoint
            schedule = [base_temp for _ in range(len(occupied_time_mask_weekly))]

            # adjust for sleep
            sleep_temp = self.assumptions.cooling_defaults.sleep_setpoint if cooling.sleep_state else cooling_max_setpoint
            schedule = OccupancyGenerator.revise_by_sleep(sleep_time_mask_weekly, schedule, sleep_temp)

            # adjust for absence
            absent_temp = self.assumptions.cooling_defaults.absent_setpoint if cooling.absent_state else cooling_max_setpoint
            schedule = OccupancyGenerator.revise_by_absence(occupied_time_mask_weekly, schedule, absent_temp)
            return schedule
        # if setpoint control
        elif cooling.type=="thermostat":
            if cooling.active_setpoint is not None:
                schedule =  [cooling.active_setpoint for _ in range(len (occupied_time_mask_weekly))]
            else:
                schedule = [cooling_max_setpoint for _ in range(len (occupied_time_mask_weekly))]
            # adjust for setback
            if cooling.sleep_setpoint is not None:
                setback = cooling.sleep_setpoint
                schedule = OccupancyGenerator.revise_by_sleep(sleep_time_mask_weekly,schedule, setback)
            else:
                schedule = OccupancyGenerator.revise_by_sleep(sleep_time_mask_weekly, schedule, cooling_max_setpoint)
            # adjust for absence
            if cooling.absent_setpoint is not None:
                setback = cooling.absent_setpoint
                schedule = OccupancyGenerator.revise_by_absence(occupied_time_mask_weekly, schedule, setback)
            else:
                schedule = OccupancyGenerator.revise_by_absence(occupied_time_mask_weekly, schedule, cooling_max_setpoint)
            return schedule
        else:
            raise NotImplementedError(f"Cooling type {cooling.type} not yet implemented.")
        

