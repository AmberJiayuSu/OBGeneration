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
        self.laundry_schedule = None
        self.fridge_schedule = None
        self.dishwasher_schedule = None

    def get_equipment_usage_schedule(self) -> dict[str, list[float]]:
        """ Translates equipment usage schedule into a full week schedule with power."""
        laundry_schedule = self.get_laundry_usage_schedule() if self.laundry_schedule is None else self.laundry_schedule
        fridge_schedule = self.get_fridge_usage_schedule() if self.fridge_schedule is None else self.fridge_schedule
        dishwasher_schedule = self.get_dishwasher_usage_schedule() if self.dishwasher_schedule is None else self.dishwasher_schedule
        sum = np.array(laundry_schedule) + np.array(fridge_schedule) + np.array(dishwasher_schedule)
        equipment_schedule = sum.tolist()
        return equipment_schedule

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
                self.laundry_schedule = [0.0] * (24 * 7)
                return self.laundry_schedule
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
                    if start_time is None:
                        self.laundry_schedule = laundry_schedule
                        return self.laundry_schedule
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
                    start_time_dist.update_weights_by_factor(update_weighted_prob)
                self.laundry_schedule = laundry_schedule
                return self.laundry_schedule
        else:
            self.laundry_schedule = [0.0] * (24 * 7)
            return self.laundry_schedule

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
            self.fridge_schedule = fridge_schedule
            return self.fridge_schedule
        else:
            self.fridge_schedule = [0.0] * (24 * 7)
            return self.fridge_schedule
        
    def _get_dishwasher_power (self) -> float:
        """ Calculates dishwasher equipment power density in W."""
        efficient_dishwasher_power = TranslationRule.RuleSet.normal_distribution_rule()(1200.0, 100.0)
        inefficient_dishwasher_power = TranslationRule.RuleSet.normal_distribution_rule()(1800.0, 150.0)
        dishwasher = self.equipment.dishwasher
        if dishwasher.has_dishwasher:
            if dishwasher.dishwasher_efficient:
                dishwasher_power = efficient_dishwasher_power.sample()
            else:
                dishwasher_power = inefficient_dishwasher_power.sample()
            return dishwasher_power
        else:
            return 0.0
        
    def _get_dishwasher_valid_start_times(self, duration: int) -> list[int]:
        # Find all valid start times where the entire duration fits in occupied time
        valid_starts = []
        for start_hour in range(24 * 7):
            # Check if all hours from start to start+duration are occupied
            all_occupied = True
            for h in range(duration):
                hour_idx = (start_hour + h) % (24 * 7)
                if not self.occupied_time_mask[hour_idx]:
                    all_occupied = False
                    break
            if all_occupied:
                valid_starts.append(start_hour)

        if not valid_starts:
            return None
        return valid_starts
        
    def get_dishwasher_usage_schedule(self) -> list[float]:
        """ Translates dishwasher equipment usage schedule into a full week schedule with power."""
        dishwasher = self.equipment.dishwasher
        if dishwasher.has_dishwasher:
            dishwasher_power = self._get_dishwasher_power()
            
            num_cycle_dist = TranslationRule.RuleSet.uniform_distribution_rule()(dishwasher.usage_frequency_per_week.min, dishwasher.usage_frequency_per_week.max)
            num_cycles = int(num_cycle_dist.sample())
            cycle_duration = 2  # assuming fixed 2-hour cycle for dishwasher
            
            valid_start_times = self._get_dishwasher_valid_start_times(cycle_duration)
            if valid_start_times is None:
                valid_start_times = self._get_dishwasher_valid_start_times(0)

            dishwasher_schedule = [0.0] * (24 * 7)
            
            if len(valid_start_times) == 0:
                self.dishwasher_schedule = dishwasher_schedule
                return self.dishwasher_schedule
            else:
                # more weights for evening times, moderate for after lunch, less for the rest, valid times only
                # more weights for friday evening and weekends
                weighted_prob = {
                    3.0: [],
                    2.0: [],
                    1.5: [],
                    1.0: []
                }
                for s in valid_start_times:
                    day_of_week, hour_of_day = Utils.hour_of_day(s)
                    if day_of_week in [5, 6]:  # Saturday, Sunday
                        if hour_of_day >= 18:
                            weighted_prob[3.0].append(s)
                        elif 12 <= hour_of_day < 16:
                            weighted_prob[2.0].append(s)
                        else:
                            weighted_prob[1.5].append(s)
                    if day_of_week == 4:  # Friday
                        if hour_of_day >= 18:
                            weighted_prob[3.0].append(s)
                        elif 12 <= hour_of_day < 16:
                            weighted_prob[1.5].append(s)
                        else:
                            weighted_prob[1.0].append(s)
                    else:  # Weekdays
                        if hour_of_day >= 19:
                            weighted_prob[2.0].append(s)
                        elif 12 <= hour_of_day < 14:
                            weighted_prob[1.5].append(s)
                        else:
                            weighted_prob[1.0].append(s)
                start_time_dist = TranslationRule.RuleSet.weighted_value_distribution_rule()(weighted_prob)
                for _ in range(num_cycles):
                    start_time = start_time_dist.sample()
                    if start_time is None:
                        self.dishwasher_schedule = dishwasher_schedule
                        return self.dishwasher_schedule
                    for h in range(cycle_duration):
                        hour_idx = (start_time + h) % (24 * 7)
                        dishwasher_schedule[hour_idx] = dishwasher_power
                    # update weighted prob to avoid overlapping cycles
                    update_weighted_prob = { 
                        0.0: [s for s in valid_start_times if start_time <= s < start_time + cycle_duration],
                        0.2: [s for s in valid_start_times if not (start_time <= s < start_time + cycle_duration) and (abs(s - start_time) < 2 or abs(s - start_time - cycle_duration) <2) ],
                    }

                    start_time_dist.update_weights_by_factor(update_weighted_prob)
                self.dishwasher_schedule = dishwasher_schedule
                return self.dishwasher_schedule
        else:
            self.dishwasher_schedule = [0.0] * (24 * 7)
            return self.dishwasher_schedule