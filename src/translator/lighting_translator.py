from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.lighting as Lighting
from translator.occupancy_translator_old import OccupancyTranslator, OccupancyTranslatorUtils
import stochastic.translation_rule as TranslationRule


class LightingTranslator:

    def __init__(self, lighting: Lighting.Lighting):
        self.lighting = lighting


    def translate_lighting_schedule_annual(self, occupancy_annual: list[list[float]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates lighting usage pattern into a full week schedule based on occupancy and sleep times."""
        assert len(occupancy_annual) == len(sleep_mask_annual), "occupancy and sleep must have same #weeks"
        assert len(occupancy_annual) == 53, "expected 53 chunks (52 weeks + 24h)"

        schedule: list[list[float]] = []
        for weekly_occupancy, weekly_sleep_mask in zip(occupancy_annual, sleep_mask_annual):
            assert len(weekly_occupancy) == len(weekly_sleep_mask), "weekly occupancy/sleep length mismatch"
            schedule.append(self.translate_lighting_schedule_weekly(weekly_occupancy, weekly_sleep_mask))
        return schedule
    
    def translate_lighting_schedule_weekly(self, occupancy_weekly: list[float], sleep_mask_weekly: list[bool]) -> list[float]:
        """ Translates lighting usage pattern into a full week schedule based on occupancy and sleep times."""
        lighting = self.lighting
        schedule = []
        length = len(occupancy_weekly)
        #full on always
        if not lighting.turn_off_habits.when_house_empty:
            schedule = [1.0] * length
        # assume full on only when occupied
        elif not lighting.turn_off_habits.when_room_empty:
            schedule = [1.0 if occ > 0.0 else 0.0 for occ in occupancy_weekly]
        # assume mannually adjusted based on occupancy
        else:
            schedule = occupancy_weekly.copy()
        # Adjust for sleep times 
        # Assumption: during sleep time, lighting usage is zero
        schedule = OccupancyTranslatorUtils.revise_by_sleep(sleep_mask_weekly, schedule, 0.0)       
        return schedule
        

    def get_dimming(self) -> bool:
        """ Determines if dimming is used based on usage pattern. """
        if self.lighting.turn_off_habits.when_daylight_bright:
            return True
        return False