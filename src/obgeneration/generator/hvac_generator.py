
from obgeneration.generator.occupancy_generator import OccupancyGenerator, HouseholdOccupancyFractions
import obgeneration.model.hvac as HVAC
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
    """Default setpoints for cooling thermostat control."""
    active_setpoint: float = Field(23.0, description="Active cooling setpoint in Celsius")
    sleep_setpoint: float = Field(22.0, description="Sleep cooling setpoint in Celsius")
    absent_setpoint: float = Field(24.0, description="Absent cooling setpoint in Celsius")


class HVACAssumptions(BaseModel):
    """Master configuration for HVAC behavioral assumptions."""
    trv: TRVAssumptions
    minimum_heating_setpoint: float = Field(10.0, description="Minimum heating setpoint in Celsius")
    maximum_cooling_setpoint: float = Field(40.0, description="Maximum cooling setpoint in Celsius")
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

    @staticmethod
    def generate_with_defaults(
        hvac: HVAC.HVAC,
        occupancy_states: list[list[HouseholdOccupancyFractions]]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Generate annual heating and cooling setpoint schedules from occupancy states using default assumptions."""
        assumptions = HVACAssumptions.default()
        generator = HVACGenerator(hvac, assumptions)
        active_mask, sleep_mask = OccupancyGenerator.active_sleep_mask(occupancy_states, 0.3)
        heating_schedule = generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling_schedule = generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)
        return heating_schedule, cooling_schedule

    @staticmethod
    def _get_trv_temp(assumptions: HVACAssumptions,level: HVAC.IntensityLevel) -> float:
        """Helper method to map intensity level to temperature."""
        trv = assumptions.trv
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

    def heating_setpoint_annual_schedule(self, active_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        if self.hvac.heating is None:
            return None
        
        assert len(active_mask_annual) == len(sleep_mask_annual), "active and sleep must have same #weeks"
        assert len(active_mask_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_active_mask, weekly_sleep_mask in zip(active_mask_annual, sleep_mask_annual):
            assert len(weekly_active_mask) == len(weekly_sleep_mask), "weekly active/sleep length mismatch"
            schedule.append(self.heating_setpoint_weekly_schedule(weekly_active_mask, weekly_sleep_mask))
        return schedule
        
       
    def heating_setpoint_weekly_schedule(self, active_mask_weekly: list[bool], sleep_mask_weekly: list[bool]) -> list[float]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        heating = self.hvac.heating
        minimum_setpoint = self.assumptions.minimum_heating_setpoint
        #if no control at all.
        if heating.type=="no_control":
            schedule = [self.assumptions.heating_defaults.active_setpoint for _ in range(len (active_mask_weekly))] # default  always on
            return schedule
        
        #if binary control
        elif heating.type=="binary":
            # start with absent_state
            absent_temp = self.assumptions.heating_defaults.absent_setpoint if heating.absent_state else minimum_setpoint
            schedule = [absent_temp for _ in range(len(active_mask_weekly))]

             # adjust for sleep
            sleep_temp = self.assumptions.heating_defaults.sleep_setpoint if heating.sleep_state else minimum_setpoint
            schedule = OccupancyGenerator.apply_to_mask(sleep_mask_weekly, schedule, sleep_temp)

            # adjust for active
            active_temp = self.assumptions.heating_defaults.active_setpoint if heating.active_state else minimum_setpoint
            schedule = OccupancyGenerator.apply_to_mask(active_mask_weekly, schedule, active_temp)

            return schedule
        
        # if valve control
        elif heating.type=="valve":
            # start with absent level
            base_temp = HVACGenerator._get_trv_temp(self.assumptions, heating.absent_level)
            schedule = [base_temp for _ in range(len(active_mask_weekly))]

            # adjust for sleep
            sleep_temp = HVACGenerator._get_trv_temp(self.assumptions, heating.sleep_level)
            schedule = OccupancyGenerator.apply_to_mask(sleep_mask_weekly, schedule, sleep_temp)

             # adjust for active level
            active_temp = HVACGenerator._get_trv_temp(self.assumptions, heating.active_level)
            schedule = OccupancyGenerator.apply_to_mask(active_mask_weekly, schedule, active_temp)

            return schedule
       
        # if setpoint control
        elif heating.type=="thermostat":
            if heating.absent_setpoint is not None:
                schedule =  [heating.absent_setpoint for _ in range(len (active_mask_weekly))]
            else:
                schedule = [minimum_setpoint for _ in range(len (active_mask_weekly))]
            # adjust for sleep
            if heating.sleep_setpoint is not None:
                setback = heating.sleep_setpoint
                schedule = OccupancyGenerator.apply_to_mask(sleep_mask_weekly, schedule, setback)
            else:
                schedule = OccupancyGenerator.apply_to_mask(sleep_mask_weekly, schedule, minimum_setpoint)
            # adjust for active
            if heating.active_setpoint is not None:
                active_temp = heating.active_setpoint
                schedule = OccupancyGenerator.apply_to_mask(active_mask_weekly, schedule, active_temp)
            else:
                schedule = OccupancyGenerator.apply_to_mask(active_mask_weekly, schedule, minimum_setpoint)
           
            return schedule
        else:
            raise NotImplementedError(f"Heating type {heating.type} not yet implemented.")



    def cooling_setpoint_annual_schedule(self, active_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        if self.hvac.cooling is None:
            return None
        
        assert len(active_mask_annual) == len(sleep_mask_annual), "active and sleep must have same #weeks"
        assert len(active_mask_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_active_mask, weekly_sleep_mask in zip(active_mask_annual, sleep_mask_annual):
            assert len(weekly_active_mask) == len(weekly_sleep_mask), "weekly active/sleep length mismatch"
            schedule.append(self.cooling_setpoint_weekly_schedule(weekly_active_mask, weekly_sleep_mask))
        return schedule


    def cooling_setpoint_weekly_schedule(self, active_time_mask_weekly: list[bool], sleep_time_mask_weekly: list[bool]) -> list[float]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        cooling = self.hvac.cooling
        cooling_max_setpoint = self.assumptions.maximum_cooling_setpoint
        #if no control at all.
        if cooling.type=="no_control":
            schedule = [self.assumptions.cooling_defaults.active_setpoint for _ in range(len (active_time_mask_weekly))]
            return schedule
        #if binary control
        elif cooling.type=="binary":
            # start with absent_state
            base_temp = self.assumptions.cooling_defaults.absent_setpoint if cooling.absent_state else cooling_max_setpoint
            schedule = [base_temp for _ in range(len(active_time_mask_weekly))]

            # adjust for sleep
            sleep_temp = self.assumptions.cooling_defaults.sleep_setpoint if cooling.sleep_state else cooling_max_setpoint
            schedule = OccupancyGenerator.apply_to_mask(sleep_time_mask_weekly, schedule, sleep_temp)

            # adjust for active
            active_temp = self.assumptions.cooling_defaults.active_setpoint if cooling.active_state else cooling_max_setpoint
            schedule = OccupancyGenerator.apply_to_mask(active_time_mask_weekly, schedule, active_temp)

            return schedule
        
        # if setpoint control
        elif cooling.type=="thermostat":
            # start with absent setpoint
            if cooling.absent_setpoint is not None:
                schedule = [cooling.absent_setpoint for _ in range(len (active_time_mask_weekly))]
            else:
                schedule = [cooling_max_setpoint for _ in range(len (active_time_mask_weekly))]

            # adjust for setback
            if cooling.sleep_setpoint is not None:
                setback = cooling.sleep_setpoint
                schedule = OccupancyGenerator.apply_to_mask(sleep_time_mask_weekly,schedule, setback)
            else:
                schedule = OccupancyGenerator.apply_to_mask(sleep_time_mask_weekly, schedule, cooling_max_setpoint)

            # adjust for active
            if cooling.active_setpoint is not None:
                active_temp = cooling.active_setpoint
                schedule = OccupancyGenerator.apply_to_mask(active_time_mask_weekly, schedule, active_temp)
            else:
                schedule = OccupancyGenerator.apply_to_mask(active_time_mask_weekly, schedule, cooling_max_setpoint)
            return schedule
        else:
            raise NotImplementedError(f"Cooling type {cooling.type} not yet implemented.")
        

