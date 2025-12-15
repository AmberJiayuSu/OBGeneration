from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.lighting as Lighting
from translator.occupancy_translator import OccupancyTranslator, OccupancyTranslatorUtils
import stochastic.translation_rule as TranslationRule


class LightingTranslator:

    def __init__(self, lighting: Lighting.Lighting):
        self.lighting = lighting

    # Maybe this should be included as part of occupant behavior, this is more like a building feature
    def get_lighting_power_density(self) -> float:
        """ Calculates lighting power density based on LED status. (W/sqm) """
        LED_presumption_dist = TranslationRule.RuleSet.normal_distribution_rule()(7.0, 1.0) 
        nonLED_presumption_dist = TranslationRule.RuleSet.normal_distribution_rule()(15.0, 2.0)
        if self.lighting.LED:
            return LED_presumption_dist.sample()
        else:
            return nonLED_presumption_dist.sample()


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
        # assume always full on even when not occupied
        if lighting.usage_pattern == Lighting.LightingBehavior.ALWAYS_ON:
            schedule = [1.0] * length
        # assume full on only when occupied
        elif lighting.usage_pattern == Lighting.LightingBehavior.MOSTLY_ON:
            for hour in range(length):
                if occupancy_weekly[hour] > 0:
                    schedule.append(1.0)
                else:
                    schedule.append(0.0)
        # assume partially on (adjusted by the occupancy level) when occupied
        else:
            schedule = occupancy_weekly.copy()
        # Adjust for sleep times 
        # Assumption: during sleep time, lighting usage is zero
        schedule = OccupancyTranslatorUtils.revise_by_sleep(sleep_mask_weekly, schedule, 0.0)       
        return schedule
        
    @staticmethod
    def get_dimming(lighting: Lighting.Lighting) -> bool:
        """ Determines if dimming is used based on usage pattern. """
        if lighting.usage_pattern != Lighting.LightingBehavior.ALWAYS_ON:
            return True
        return False