from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.hvac as HVAC
from translator.occupancy_translator_old import OccupancyTranslator,OccupancyTranslatorUtils
import stochastic.translation_rule as TranslationRule



class HVACTranslator:
    def __init__(self, hvac: HVAC.HVAC):
        self.hvac = hvac

    def translate_heating_setpoint_schedule_annually(self, occupancy_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        if self.hvac.heating is None:
            return None
        
        assert len(occupancy_mask_annual) == len(sleep_mask_annual), "occupancy and sleep must have same #weeks"
        assert len(occupancy_mask_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_occupancy, weekly_sleep_mask in zip(occupancy_mask_annual, sleep_mask_annual):
            assert len(weekly_occupancy) == len(weekly_sleep_mask), "weekly occupancy/sleep length mismatch"
            schedule.append(self.translate_heating_setpoint_schedule_weekly(weekly_occupancy, weekly_sleep_mask))
        return schedule
        
       
    def translate_heating_setpoint_schedule_weekly(self, occupancy_mask_weekly: list[bool], sleep_mask_weekly: list[bool]) -> list[float]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        heating = self.hvac.heating
        #if no control at all.
        if heating.type=="no_control":
            schedule = [20.0 for _ in range(len (occupancy_mask_weekly))] # default 20C always on
            return schedule
        # if setpoint control
        elif heating.type=="thermostat":
            if heating.active_setpoint is not None:
                schedule =  [heating.active_setpoint for _ in range(len (occupancy_mask_weekly))]
            else:
                schedule = [0.0 for _ in range(len (occupancy_mask_weekly))]
            # adjust for setback
            if heating.sleep_setpoint is not None:
                setback = heating.sleep_setpoint
                schedule = OccupancyTranslatorUtils.revise_by_sleep(sleep_mask_weekly,schedule, setback)
            else:
                schedule = OccupancyTranslatorUtils.revise_by_sleep(sleep_mask_weekly, schedule, 0.0)
            # adjust for absence
            if heating.absent_setpoint is not None:
                setback = heating.absent_setpoint
                schedule = OccupancyTranslatorUtils.revise_by_absence(occupancy_mask_weekly, schedule, setback)
            else:
                schedule = OccupancyTranslatorUtils.revise_by_absence(occupancy_mask_weekly, schedule, 0.0)
            return schedule
        else:
            raise NotImplementedError(f"Heating type {heating.type} not yet implemented.")



    def translate_cooling_setpoint_schedule_annually(self, occupancy_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        if self.hvac.cooling is None:
            return None
        
        assert len(occupancy_mask_annual) == len(sleep_mask_annual), "occupancy and sleep must have same #weeks"
        assert len(occupancy_mask_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_occupancy, weekly_sleep_mask in zip(occupancy_mask_annual, sleep_mask_annual):
            assert len(weekly_occupancy) == len(weekly_sleep_mask), "weekly occupancy/sleep length mismatch"
            schedule.append(self.translate_cooling_setpoint_schedule_weekly(weekly_occupancy, weekly_sleep_mask))
        return schedule


    def translate_cooling_setpoint_schedule_weekly(self, occupied_time_mask_weekly: list[bool], sleep_time_mask_weekly: list[bool]) -> list[float]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        cooling = self.hvac.cooling
        #if no control at all.
        if cooling.type=="no_control":
            schedule = [24.0 for _ in range(len (occupied_time_mask_weekly))] # default 24C always on
            return schedule
        # if setpoint control
        elif cooling.type=="thermostat":
            if cooling.active_setpoint is not None:
                schedule =  [cooling.active_setpoint for _ in range(len (occupied_time_mask_weekly))]
            else:
                schedule = [35.0 for _ in range(len (occupied_time_mask_weekly))]
            # adjust for setback
            if cooling.sleep_setpoint is not None:
                setback = cooling.sleep_setpoint
                schedule = OccupancyTranslatorUtils.revise_by_sleep(sleep_time_mask_weekly,schedule, setback)
            else:
                schedule = OccupancyTranslatorUtils.revise_by_sleep(sleep_time_mask_weekly, schedule, 35.0)
            # adjust for absence
            if cooling.absent_setpoint is not None:
                setback = cooling.absent_setpoint
                schedule = OccupancyTranslatorUtils.revise_by_absence(occupied_time_mask_weekly, schedule, setback)
            else:
                schedule = OccupancyTranslatorUtils.revise_by_absence(occupied_time_mask_weekly, schedule, 35.0)
            return schedule
        else:
            raise NotImplementedError(f"Cooling type {cooling.type} not yet implemented.")
        

