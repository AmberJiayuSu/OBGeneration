from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.equipment as Equipment
from stochastic.occupancy_translator import OccupancyTranslator
import stochastic.translation_rule as TranslationRule
from stochastic.translator_utils import Utils


class EquipmentTranslator:
    def __init__(self, equipment: Equipment.Equipment, occupancy_translator,):
        self.equipment = equipment
        self.active_time_mask = occupancy_translator.get_occupied_active_mask()
        self.occupied_time_mask = occupancy_translator.get_occupied_mask()

    def _get_laundry_power(self) -> tuple[float, float]:
        """ Calculates laundry equipment power density in W."""
        efficient_washer_power = TranslationRule.RuleSet.normal_distribution_rule()(400.0, 50.0)
        inefficient_washer_power = TranslationRule.RuleSet.normal_distribution_rule()(1600.0, 150.0)
        efficient_dryer_power = TranslationRule.RuleSet.normal_distribution_rule()(2500.0, 200.0)
        inefficient_dryer_power = TranslationRule.RuleSet.normal_distribution_rule()(4000.0, 300.0)
        washer_power = 0.0
        dryer_power = 0.0
        laundry = self.equipment.laundry
        if laundry.has_washer:
            if laundry.washer_efficient:
                washer_power = efficient_washer_power.sample()
            else:
                washer_power = inefficient_washer_power.sample()
        if laundry.has_dryer:
            if laundry.dryer_efficient:
                dryer_power = efficient_dryer_power.sample()
            else:
                dryer_power = inefficient_dryer_power.sample()
        return washer_power, dryer_power
    
    def get_laundry_usage_schedule(self) -> list[float]:
        """ Translates laundry equipment usage schedule into a full week schedule with power."""
        laundry = self.equipment.laundry
        if laundry.has_washer or laundry.has_dryer:
            washer_power, dryer_power = self._get_laundry_power()
            num_cycle_dist = TranslationRule.RuleSet.uniform_distribution_rule()(laundry.usage_frequency_per_week.min, laundry.usage_frequency_per_week.max)
            num_cycles = int(num_cycle_dist.sample())

            # Determine cycle duration based on equipment
            washer_duration = 1 if laundry.has_washer else 0
            dryer_duration = 1 if laundry.has_dryer else 0
            cycle_duration = washer_duration + dryer_duration
           
            valid_start_times = self._get_laundry_valid_start_times(cycle_duration)
            if valid_start_times is None:
                valid_start_times = self._get_laundry_valid_start_times(0)
            
            if len(valid_start_times) == 0:
                # No valid start times found
                return [0.0] * (24 * 7)
            else:
                weighted_prob = {
                    1.0: [s for s in valid_start_times if Utils.hour_of_day(s)[1] <= 19 and Utils.hour_of_day(s)[0] <= 5], 
                    1.5: [s for s in valid_start_times if 19 < Utils.hour_of_day(s)[1] <= 22 and Utils.hour_of_day(s)[0] <= 5],
                    2.0: [s for s in valid_start_times if Utils.hour_of_day(s)[0] > 5]
                }
                start_time_dist = TranslationRule.RuleSet.weighted_value_distribution_rule()(weighted_prob)
                laundry_schedule = [0.0] * (24 * 7)
                for _ in range(num_cycles):
                    start_time = start_time_dist.sample()
                    for h in range(cycle_duration):
                        hour_idx = (start_time + h) % (24 * 7)
                        if h < washer_duration:
                            laundry_schedule[hour_idx] = washer_power
                        else:
                            laundry_schedule[hour_idx] = dryer_power
                    # update weighted prob to avoid overlapping cycles, and also prefer not to not do laundry multiple times in the same day
                    update_weighted_prob = { 
                        0.0: [s for s in valid_start_times if start_time <= s < start_time + cycle_duration],
                        0.2: [s for s in valid_start_times if not (start_time <= s < start_time + cycle_duration) and Utils.hour_of_day(s)[0] == Utils.hour_of_day(start_time)[0]],
                    }
                    start_time_dist.update_weights(update_weighted_prob)
                return laundry_schedule
        else:
            return [0.0] * (24 * 7)

    def _get_laundry_valid_start_times(self, duration: int) -> list[int]:
        # Find all valid start times where the entire duration fits in active time
        valid_starts = []
        for start_hour in range(24 * 7):
            # Check if all hours from start to start+duration are active
            all_active = True
            for h in range(duration):
                hour_idx = (start_hour + h) % (24 * 7)
                if not self.active_time_mask[hour_idx]:
                    all_active = False
                    break
            if all_active:
                valid_starts.append(start_hour)

        if not valid_starts:
            return None
        return valid_starts

    def get_fridge_usage_schedule(self) -> list[float]:
        """ Translates refrigeration equipment usage schedule into a full week schedule with power."""
        refrigeration = self.equipment.refrigeration
        if refrigeration.has_refrigerator:
            size_power_map = {
                Equipment.RefrigerationSize.MINI: TranslationRule.RuleSet.normal_distribution_rule()(100.0, 25.0),
                Equipment.RefrigerationSize.SMALL: TranslationRule.RuleSet.normal_distribution_rule()(300.0, 30.0),
                Equipment.RefrigerationSize.MEDIUM: TranslationRule.RuleSet.normal_distribution_rule()(400.0, 50.0),
                Equipment.RefrigerationSize.LARGE: TranslationRule.RuleSet.normal_distribution_rule()(600.0, 75.0),
            }
            base_power_dist = size_power_map[refrigeration.size]
            base_power = base_power_dist.sample()
            
            if not refrigeration.efficient_refrigerator:
                in_efficient_dist = TranslationRule.RuleSet.bernoulli_distribution_rule()(1.3, 0.1)
                in_efficient_scaler = in_efficient_dist.sample()
                base_power *= in_efficient_scaler
            fridge_schedule = [base_power] * (24 * 7)
            return fridge_schedule
        else:
            return [0.0] * (24 * 7)
