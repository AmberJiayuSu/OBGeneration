from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.lighting as Lighting
from translator.occupancy_translator import OccupancyTranslator
import stochastic.translation_rule as TranslationRule


class LightingTranslator:

    def __init__(self, lighting: Lighting.Lighting, sleep_mask : list[bool], occupancy_weekly : list[float]):
        self.lighting = lighting
        self.sleep_time_mask = sleep_mask
        self.occupancy_weekly = occupancy_weekly

    def get_lighting_power_density(self) -> float:
        """ Calculates lighting power density based on LED status. (W/sqm) """
        LED_presumption_dist = TranslationRule.RuleSet.normal_distribution_rule()(7.0, 1.0) 
        nonLED_presumption_dist = TranslationRule.RuleSet.normal_distribution_rule()(15.0, 2.0)
        if self.lighting.LED:
            return LED_presumption_dist.sample()
        else:
            return nonLED_presumption_dist.sample()


    def translate_lighting_schedule(self) -> list[float]:
        """ Translates lighting usage pattern into a full week schedule based on occupancy and sleep times."""
        lighting = self.lighting
        schedule = []
        # assume always full on even when not occupied
        if lighting.usage_pattern == Lighting.LightingBehavior.ALWAYS_ON:
            schedule = [1.0] * 24 * 7
        # assume full on only when occupied
        elif lighting.usage_pattern == Lighting.LightingBehavior.MOSTLY_ON:
            for hour in range(24 * 7):
                if self.occupancy_weekly[hour] > 0:
                    schedule.append(1.0)
                else:
                    schedule.append(0.0)
        # assume partially on (adjusted by the occupancy level) when occupied
        else:
            for hour in range(24 * 7):
                schedule.append(self.occupancy_weekly[hour])
        # Adjust for sleep times 
        # Assumption: during sleep time, lighting usage is zero
        schedule = OccupancyTranslator.revise_by_sleep(self.sleep_time_mask, schedule, 0.0)       
        return schedule
        
    @staticmethod
    def get_dimming(lighting: Lighting.Lighting) -> bool:
        """ Determines if dimming is used based on usage pattern. """
        if lighting.usage_pattern != Lighting.LightingBehavior.ALWAYS_ON:
            return True
        return False