from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.hvac as HVAC
from stochastic.occupancy_translator import OccupancyTranslator
import stochastic.translation_rule as TranslationRule



class HVACTranslator:
    def __init__(self, hvac: HVAC.HVAC):
        self.hvac = hvac

    def translate_heating_setpoint_schedule(self, occupancy_weekly:list[float], sleep_weekly:list[tuple[int,int] | None]) -> list[float]:
        """ Translates HVAC heating setpoint schedule in celcius into a full week schedule."""
        hvac = self.hvac
        if hvac.heating.control.setpoint.setpoint_celsius is not None:
            schedule =  [self.hvac.heating.control.setpoint.setpoint_celsius] * 24 * 7 
        else:
            schedule = [0.0] * 24 * 7  # basically no heating if not specified
        # adjust for setback
        if hvac.heating.control.setpoint.setback_sleep_celsius is not None:
            setback = hvac.heating.control.setpoint.setback_sleep_celsius
            schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule, setback)  
        else:
            schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule, 0.0)
        # adjust for absence
        if hvac.heating.control.setpoint.setback_absent_celsius is not None:
            setback = hvac.heating.control.setpoint.setback_absent_celsius
            for hour in range(24 * 7):
                if occupancy_weekly[hour] == 0.0:
                    schedule[hour] = setback
        else:
            for hour in range(24 * 7):
                if occupancy_weekly[hour] == 0.0:
                    schedule[hour] = 0.0

        return schedule
    
    def translate_cooling_setpoint_schedule(self, occupancy_weekly:list[float], sleep_weekly:list[tuple[int,int] | None]) -> list[float]:
        """ Translates HVAC cooling setpoint schedule in celcius into a full week schedule."""
        hvac = self.hvac
        if hvac.cooling.control.setpoint.setpoint_celsius is not None:
            schedule =  [self.hvac.cooling.control.setpoint.setpoint_celsius] * 24 * 7
        else:
            schedule = [35.0] * 24 * 7 # basically no cooling if not specified
        # adjust for setback
        if hvac.cooling.control.setpoint.setback_sleep_celsius is not None:
            setback = hvac.cooling.control.setpoint.setback_sleep_celsius
            schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule, setback)  
        else:
            schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule, 35.0)
        # adjust for absence
        if hvac.cooling.control.setpoint.setback_absent_celsius is not None:
            setback = hvac.cooling.control.setpoint.setback_absent_celsius
            for hour in range(24 * 7):
                if occupancy_weekly[hour] == 0.0:
                    schedule[hour] = setback
        else:
            for hour in range(24 * 7):
                if occupancy_weekly[hour] == 0.0:
                    schedule[hour] = 35.0

        return schedule
    
    def translate_heating_availability_schedule(self, occupancy_weekly:list[float], sleep_weekly:list[tuple[int,int] | None]) -> list[bool]:
        """ Translates HVAC heating availability schedule into a full week schedule."""
        hvac = self.hvac
        if not hvac.heating.onoff_control:
            return [True] * 24 * 7
        else:
            if hvac.heating.control.type == "none":
                schedule = []
                for hour in range(24 * 7):
                    if occupancy_weekly[hour] > 0.0:
                        schedule.append(hvac.heating.control.onoff.active)
                    else:
                        schedule.append(hvac.heating.control.onoff.absent)
                schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule, hvac.heating.control.onoff.sleep)  
                return schedule
            elif hvac.heating.control.type == "valve":
                schedule = []
                for hour in range(24 * 7):
                    if occupancy_weekly[hour] > 0.0:
                        if hvac.heating.control.valve.heating_valve_active_percentage is not None and hvac.heating.control.valve.heating_valve_active_percentage > 0:
                            schedule.append(True)
                        else:
                            schedule.append(False)
                    else:
                        if hvac.heating.control.valve.heating_valve_absent_percentage is not None and hvac.heating.control.valve.heating_valve_absent_percentage > 0:
                            schedule.append(True)
                        else:
                            schedule.append(False)
                sleep_cond = hvac.heating.control.valve.heating_valve_sleep_percentage is not None and hvac.heating.control.valve.heating_valve_sleep_percentage > 0
                schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule,  sleep_cond)  
                return schedule
            else:
                schedule = []
                for hour in range(24 * 7):
                    if occupancy_weekly[hour] > 0.0:
                        if hvac.heating.control.setpoint.setpoint_celsius is not None:
                            schedule.append(True)
                        else:
                            schedule.append(False)
                    else:
                        if hvac.heating.control.setpoint.setback_absent_celsius is not None:
                            schedule.append(True)
                        else:
                            schedule.append(False)
                sleep_cond = hvac.heating.control.setpoint.setback_sleep_celsius is not None
                schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule,  sleep_cond)
                return schedule

            
    def translate_cooling_availability_schedule(self, occupancy_weekly:list[float], sleep_weekly:list[tuple[int,int] | None]) -> list[bool]:
        """ Translates HVAC cooling availability schedule into a full week schedule."""
        hvac = self.hvac
        if not hvac.cooling.onoff_control:
            return [True] * 24 * 7
        else:
            if hvac.cooling.control.type == "none":
                schedule = []
                for hour in range(24 * 7):
                    if occupancy_weekly[hour] > 0.0:
                        schedule.append(hvac.cooling.control.onoff.active)
                    else:
                        schedule.append(hvac.cooling.control.onoff.absent)
                schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule, hvac.cooling.control.onoff.sleep)  
                return schedule
            elif hvac.cooling.control.type == "thermostat_setpoint":
                schedule = []
                for hour in range(24 * 7):
                    if occupancy_weekly[hour] > 0.0:
                        if hvac.cooling.control.setpoint.setpoint_celsius is not None:
                            schedule.append(True)
                        else:
                            schedule.append(False)
                    else:
                        if hvac.cooling.control.setpoint.setback_absent_celsius is not None:
                            schedule.append(True)
                        else:
                            schedule.append(False)
                sleep_cond = hvac.cooling.control.setpoint.setback_sleep_celsius is not None
                schedule = OccupancyTranslator.revise_by_sleep(sleep_weekly, schedule,  sleep_cond)  
                return schedule