from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.hvac as HVAC
from translator.occupancy_translator import OccupancyTranslator
import stochastic.translation_rule as TranslationRule



class HVACTranslator:
    def __init__(self, hvac: HVAC.HVAC, sleep_mask :list[bool], occupied_mask :list[bool]):
        self.hvac = hvac
        self.sleep_time_mask = sleep_mask
        self.occupied_time_mask = occupied_mask

    def translate_heating_setpoint_schedule(self) -> list[float]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        hvac = self.hvac
        if hvac.heating.control.setpoint.setpoint_celsius is not None:
            schedule =  [self.hvac.heating.control.setpoint.setpoint_celsius] * 24 * 7 
        else:
            schedule = [0.0] * 24 * 7  # basically no heating if not specified
        # adjust for setback
        if hvac.heating.control.setpoint.setback_sleep_celsius is not None:
            setback = hvac.heating.control.setpoint.setback_sleep_celsius
            schedule = OccupancyTranslator.revise_by_sleep(self.sleep_time_mask, schedule, setback)  
        else:
            schedule = OccupancyTranslator.revise_by_sleep(self.sleep_time_mask, schedule, 0.0)
        # adjust for absence
        if hvac.heating.control.setpoint.setback_absent_celsius is not None:
            setback = hvac.heating.control.setpoint.setback_absent_celsius
            schedule = OccupancyTranslator.revise_by_absence(self.occupied_time_mask, schedule, setback)
        else:
            schedule = OccupancyTranslator.revise_by_absence(self.occupied_time_mask, schedule, 0.0)

        return schedule
    
    def translate_cooling_setpoint_schedule(self) -> list[float]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        hvac = self.hvac
        if hvac.cooling.control.setpoint.setpoint_celsius is not None:
            schedule =  [self.hvac.cooling.control.setpoint.setpoint_celsius] * 24 * 7
        else:
            schedule = [35.0] * 24 * 7 # basically no cooling if not specified
        # adjust for setback
        if hvac.cooling.control.setpoint.setback_sleep_celsius is not None:
            setback = hvac.cooling.control.setpoint.setback_sleep_celsius
            schedule = OccupancyTranslator.revise_by_sleep(self.sleep_time_mask,schedule, setback)  
        else:
            schedule = OccupancyTranslator.revise_by_sleep(self.sleep_time_mask, schedule, 35.0)
        # adjust for absence
        if hvac.cooling.control.setpoint.setback_absent_celsius is not None:
            setback = hvac.cooling.control.setpoint.setback_absent_celsius
            schedule = OccupancyTranslator.revise_by_absence(self.occupied_time_mask, schedule, setback)
        else:
            schedule = OccupancyTranslator.revise_by_absence(self.occupied_time_mask, schedule, 35.0)

        return schedule
    