from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.equipment as Equipment
from stochastic.occupancy_translator import OccupancyTranslator
import stochastic.translation_rule as TranslationRule
from stochastic.translator_utils import Utils


class EquipmentTranslator:
    system_base_load = 1000.0  # W, base load for equipment not modeled explicitly
    laundry_power_map = {
        "efficient_washer": TranslationRule.RuleSet.normal_distribution_rule()(400.0, 50.0),
        "inefficient_washer": TranslationRule.RuleSet.normal_distribution_rule()(1600.0, 150.0),
        "efficient_dryer": TranslationRule.RuleSet.normal_distribution_rule()(2500.0, 200.0),
        "inefficient_dryer": TranslationRule.RuleSet.normal_distribution_rule()(4000.0, 300.0),
    }
    laundry_duration_dist = TranslationRule.RuleSet.normal_distribution_rule()(2.5, 0.5, context={"lower":1, "upper":4})
    refrigeration_power_map = {
        Equipment.RefrigerationSize.MINI: TranslationRule.RuleSet.normal_distribution_rule()(100.0, 25.0),
        Equipment.RefrigerationSize.SMALL: TranslationRule.RuleSet.normal_distribution_rule()(300.0, 30.0),
        Equipment.RefrigerationSize.MEDIUM: TranslationRule.RuleSet.normal_distribution_rule()(400.0, 50.0),
        Equipment.RefrigerationSize.LARGE: TranslationRule.RuleSet.normal_distribution_rule()(600.0, 75.0),
    }
    refrigeration_efficiency_dist = TranslationRule.RuleSet.normal_distribution_rule()(1.3, 0.1)
    dishwasher_power_map = {
        "efficient_dishwasher": TranslationRule.RuleSet.normal_distribution_rule()(1200.0, 100.0),
        "inefficient_dishwasher": TranslationRule.RuleSet.normal_distribution_rule()(1800.0, 150.0),
    }
    dishwasher_cycle_duration_dist = TranslationRule.RuleSet.normal_distribution_rule()(1.5, 0.3, context={"lower":1, "upper":3}, int = True)
    kitchen_power_dist_map ={
        "all_elec_efficient": TranslationRule.RuleSet.normal_distribution_rule()(1000.0, 200.0),
        "all_elec_inefficient": TranslationRule.RuleSet.normal_distribution_rule()(3000.0, 400.0),
        "notallelec_forelec_efficient": TranslationRule.RuleSet.normal_distribution_rule()(1000.0, 100.0),
        "notallelec_forelec_inefficient": TranslationRule.RuleSet.normal_distribution_rule()(2000.0, 200.0),
        "gas_cooktop_efficient": TranslationRule.RuleSet.normal_distribution_rule()(800.0, 100.0),
        "gas_cooktop_inefficient": TranslationRule.RuleSet.normal_distribution_rule()(1200.0, 150.0),
    }
    kitchen_duration_dist = TranslationRule.RuleSet.normal_distribution_rule()(2.0, 0.5, context={"lower":1, "upper":4})
    kitchen_original_weights = {
        3.0: [s for s in range(24 * 7) if (17 <= Utils.hour_of_day(s)[1] < 21) and Utils.hour_of_day(s)[0] in [5,6]],  # weekend dinner 
        2.0: [s for s in range(24 * 7) if (17 <= Utils.hour_of_day(s)[1] < 21) and Utils.hour_of_day(s)[0] not in [5,6]] # weekday dinner
        + [s for s in range(24 * 7) if (11 <= Utils.hour_of_day(s)[1] < 14) and Utils.hour_of_day(s)[0] in [5,6]],  # weekend lunch
        1.5: [s for s in range(24 * 7) if (11 <= Utils.hour_of_day(s)[1] < 14) and Utils.hour_of_day(s)[0] not in [5,6]],  # weekday lunch
        1.0: [s for s in range(24 * 7) if (7 <= Utils.hour_of_day(s)[1] < 10) ],  # breakfast
        0.2: [s for s in range(24 * 7) if s not in (
            [s for s in range(24 * 7) if (17 <= Utils.hour_of_day(s)[1] < 21)],  #dinner times
            [s for s in range(24 * 7) if (11 <= Utils.hour_of_day(s)[1] < 14)],  # lunch times
            [s for s in range(24 * 7) if (7 <= Utils.hour_of_day(s)[1] < 10)],  # breakfast times
        )]  # other times
    }

    #TODO: put other weighted prob for start times here

    def __init__(self, equipment: Equipment.Equipment, active_mask : list[bool]):
        self.equipment = equipment
        self.active_time_mask = active_mask
        self.laundry_schedule = None
        self.fridge_schedule = None
        self.dishwasher_schedule = None
        self.kitchen_electric_schedule = None
        self.kitchen_gas_schedule = None
        

    def get_equipment_usage_schedule(self) -> dict[str, list[float]]:
        """ Translates equipment usage schedule into a full week schedule with power."""
        laundry_schedule = self.get_laundry_usage_schedule() if self.laundry_schedule is None else self.laundry_schedule
        fridge_schedule = self.get_fridge_usage_schedule() if self.fridge_schedule is None else self.fridge_schedule
        dishwasher_schedule = self.get_dishwasher_usage_schedule() if self.dishwasher_schedule is None else self.dishwasher_schedule
        electric_kitchen,gas_kitchen = self.get_kitchen_usage_schedule() if self.kitchen_electric_schedule is None or self.kitchen_gas_schedule is None else (self.kitchen_electric_schedule, self.kitchen_gas_schedule)
        base_load = [ EquipmentTranslator.system_base_load] * (24 * 7)
        sum = np.array(laundry_schedule) + np.array(fridge_schedule) + np.array(dishwasher_schedule) + np.array(electric_kitchen) + np.array(base_load)
        equipment_schedule = sum.tolist()
        return equipment_schedule
    
    def _get_valid_start_times(self, duration: int) -> list[int]:
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

    def _get_laundry_power(self) -> tuple[float, float]:
        """ Calculates laundry equipment power density in W."""
        washer_power = 0.0
        dryer_power = 0.0
        laundry = self.equipment.laundry
        if laundry.has_washer:
            if laundry.washer_efficient:
                washer_power = EquipmentTranslator.laundry_power_map["efficient_washer"].sample()
            else:
                washer_power = EquipmentTranslator.laundry_power_map["inefficient_washer"].sample()
        if laundry.has_dryer:
            if laundry.dryer_efficient:
                dryer_power = EquipmentTranslator.laundry_power_map["efficient_dryer"].sample()
            else:
                dryer_power = EquipmentTranslator.laundry_power_map["inefficient_dryer"].sample()
        return washer_power, dryer_power
    
    def get_laundry_usage_schedule(self) -> list[float]:
        """ Translates laundry equipment usage schedule into a full week schedule with power."""
        if self.laundry_schedule is not None:
            return self.laundry_schedule
        laundry = self.equipment.laundry
        if laundry.has_washer or laundry.has_dryer:
            washer_power, dryer_power = self._get_laundry_power()
            num_cycle_dist = TranslationRule.RuleSet.uniform_distribution_rule()(laundry.usage_frequency_per_week.min, laundry.usage_frequency_per_week.max)
            num_cycles = int(num_cycle_dist.sample())

            # Determine cycle duration based on equipment
            washer_duration = 1.5 if laundry.has_washer else 0
            dryer_duration = 1 if laundry.has_dryer else 0
            cycle_duration_mean = washer_duration + dryer_duration
            EquipmentTranslator.laundry_duration_dist.update_mean(cycle_duration_mean)
            cycle_duration = int(EquipmentTranslator.laundry_duration_dist.sample())
  
            valid_start_times = self._get_valid_start_times(cycle_duration)
            if valid_start_times is None:
                valid_start_times = self._get_valid_start_times(0)
            
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

    def _get_fridge_power(self) -> float:
        """ Calculates refrigeration equipment power density in W."""
        refrigeration = self.equipment.refrigeration
        if refrigeration.has_refrigerator:
            base_power_dist = EquipmentTranslator.refrigeration_power_map[refrigeration.size]
            base_power = base_power_dist.sample()
            if not refrigeration.efficient_refrigerator:
                in_efficient_scaler = EquipmentTranslator.refrigeration_efficiency_dist.sample()
                base_power *= in_efficient_scaler
            return base_power
        else:
            return 0.0
        
    def get_fridge_usage_schedule(self) -> list[float]:
        """ Translates refrigeration equipment usage schedule into a full week schedule with power."""
        if self.fridge_schedule is not None:
            return self.fridge_schedule
        refrigeration = self.equipment.refrigeration
        if refrigeration.has_refrigerator:
            base_power = self._get_fridge_power()
            fridge_schedule = [base_power] * (24 * 7)
            self.fridge_schedule = fridge_schedule
            return self.fridge_schedule
        else:
            self.fridge_schedule = [0.0] * (24 * 7)
            return self.fridge_schedule
        
    def _get_dishwasher_power (self) -> float:
        """ Calculates dishwasher equipment power density in W."""
        dishwasher = self.equipment.dishwasher
        if dishwasher.has_dishwasher:
            if dishwasher.dishwasher_efficient:
                dishwasher_power = EquipmentTranslator.dishwasher_power_map["efficient_dishwasher"].sample()
            else:
                dishwasher_power = EquipmentTranslator.dishwasher_power_map["inefficient_dishwasher"].sample()
            return dishwasher_power
        else:
            return 0.0

    def get_dishwasher_usage_schedule(self) -> list[float]:
        """ Translates dishwasher equipment usage schedule into a full week schedule with power."""
        if self.dishwasher_schedule is not None:
            return self.dishwasher_schedule
        dishwasher = self.equipment.dishwasher
        if dishwasher.has_dishwasher:
            dishwasher_power = self._get_dishwasher_power()
            
            num_cycle_dist = TranslationRule.RuleSet.uniform_distribution_rule()(dishwasher.usage_frequency_per_week.min, dishwasher.usage_frequency_per_week.max)
            num_cycles = int(num_cycle_dist.sample())
            cycle_duration = EquipmentTranslator.dishwasher_cycle_duration_dist.sample()
            
            valid_start_times = self._get_valid_start_times(cycle_duration)
            if valid_start_times is None:
                valid_start_times = self._get_valid_start_times(0)

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
        
    def _get_kitchen_electricity_dist(self) -> float:
        kitchen_equipment = self.equipment.kitchen
        if kitchen_equipment.has_kitchen_equipment:
            if kitchen_equipment.cooktop_fuel == Equipment.FuelType.ELECTRIC:
                if kitchen_equipment.efficient:
                    return EquipmentTranslator.kitchen_power_dist_map["all_elec_efficient"]
                else:
                    return EquipmentTranslator.kitchen_power_dist_map["all_elec_inefficient"]
            else:
                if kitchen_equipment.efficient:
                    return EquipmentTranslator.kitchen_power_dist_map["notallelec_forelec_efficient"]
                else:
                    return EquipmentTranslator.kitchen_power_dist_map["notallelec_forelec_inefficient"]    
        else:
            return None
        
    def _get_kitchen_gas_dist(self) -> float:
        kitchen_equipment = self.equipment.kitchen
        if kitchen_equipment.has_kitchen_equipment:
            if kitchen_equipment.cooktop_fuel == Equipment.FuelType.GAS:
                if kitchen_equipment.efficient:
                    return EquipmentTranslator.kitchen_power_dist_map["gas_cooktop_efficient"]
                else:
                    return EquipmentTranslator.kitchen_power_dist_map["gas_cooktop_inefficient"]
        return None
    
    def get_kitchen_usage_schedule(self) -> tuple[list[float], list[float]]:
        """ Translates kitchen equipment usage schedule into a full week schedule with power."""
        if self.kitchen_electric_schedule is not None and self.kitchen_gas_schedule is not None:
            return self.kitchen_electric_schedule, self.kitchen_gas_schedule
        kitchen_equipment = self.equipment.kitchen
        if kitchen_equipment.has_kitchen_equipment:
            elec_dist = self._get_kitchen_electricity_dist()
            gas_dist = self._get_kitchen_gas_dist()

            num_usage_dist = TranslationRule.RuleSet.uniform_distribution_rule()(kitchen_equipment.usage_frequency_per_week.min, kitchen_equipment.usage_frequency_per_week.max)
            num_usages = int(num_usage_dist.sample())
            duration = int(self.kitchen_duration_dist.sample())

            valid_start_times = self._get_valid_start_times(duration)
            if valid_start_times is None:
                valid_start_times = self._get_valid_start_times(0)

            kitchen_filtered_weights = {}
            for weight, time_slots in EquipmentTranslator.kitchen_original_weights.items():
                # Keep only times that are in valid_start_times
                filtered_slots = list(set(time_slots) & set(valid_start_times))
                if filtered_slots:  # Only add if there are valid times
                    kitchen_filtered_weights[weight] = filtered_slots
            start_time_dist = TranslationRule.RuleSet.weighted_value_distribution_rule()(kitchen_filtered_weights)
            elec_schedule = [0.0] * (24 * 7)
            gas_schedule = [0.0] * (24 * 7)
            for _ in range(num_usages):
                start_time = start_time_dist.sample()
                if start_time is None:
                    return self.kitchen_electric_schedule, self.kitchen_gas_schedule
                for h in range(duration):
                    hour_idx = (start_time + h) % (24 * 7)
                    if elec_dist is not None:
                        elec_schedule[hour_idx] += elec_dist.sample()
                    if gas_dist is not None:
                        gas_schedule[hour_idx] += gas_dist.sample()
                # update weighted prob to avoid overlapping cycles
                update_weighted_prob = { 
                    0.0: [s for s in valid_start_times if start_time <= s < start_time + duration],
                    0.2: [s for s in valid_start_times if not (start_time <= s < start_time + duration) and (abs(s - start_time) < 2 or abs(s - start_time - duration) <2) ],
                }
                start_time_dist.update_weights_by_factor(update_weighted_prob)
            self.kitchen_electric_schedule = elec_schedule
            self.kitchen_gas_schedule = gas_schedule
            return self.kitchen_electric_schedule, self.kitchen_gas_schedule
        else:
            elec_schedule = [0.0] * (24 * 7)
            gas_schedule = [0.0] * (24 * 7)
            self.kitchen_electric_schedule = elec_schedule
            self.kitchen_gas_schedule = gas_schedule
            return self.kitchen_electric_schedule, self.kitchen_gas_schedule