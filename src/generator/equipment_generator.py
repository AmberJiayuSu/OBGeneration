from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.equipment as Equipment
from pydantic import BaseModel, Field, ConfigDict
from stochastic.distribution_config import DistributionConfig
import json
from pathlib import Path
import csv
import random

class EventAssumptions(BaseModel):
    """Event timing assumptions for different activities of a typical week (from monday to sunday)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    start_time_probabilities: list[float] = Field(..., description="Probability distribution for event start times ")
    

    @classmethod
    def from_csv_file(cls, path: str, resolution_mins: int = 15) -> "EventAssumptions":
        """Load laundry event assumptions from a CSV file with columns: minute_of_day, probability, day_of_week, time_label"""
        csv_path = Path(path)
        probabilities = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                probabilities.append(float(row['probability']))
        event_assumptions = cls(start_time_probabilities=probabilities)
        event_assumptions.update_resolution(resolution_mins)
        return event_assumptions

    def update_resolution(self, resolution_mins: int) -> None:
        """Aggregate probabilities from 5-minute bins to target resolution and update in place.
        Args:
            resolution_mins: Target resolution in minutes (must be multiple of 5)
        """
        if resolution_mins == 5:
            return
        if resolution_mins % 5 != 0:
            raise ValueError(f"Resolution should be multiple of 5 as implemented now, got {resolution_mins}")

        aggregation_factor = resolution_mins // 5
        aggregated = []
        for i in range(0, len(self.start_time_probabilities), aggregation_factor):
            bin_sum = sum(self.start_time_probabilities[i:i+aggregation_factor])
            aggregated.append(bin_sum)

        self.start_time_probabilities = aggregated
    


class LaundryAssumptions(BaseModel):
    """Power, Duration assumptions for laundry equipment in Watts."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    efficient_washer: Distribution = Field(..., description="Distribution for efficient washer power (W)")
    inefficient_washer: Distribution = Field(..., description="Distribution for inefficient washer power (W)")
    efficient_dryer: Distribution = Field(..., description="Distribution for efficient dryer power (W)")
    inefficient_dryer: Distribution = Field(..., description="Distribution for inefficient dryer power (W)")
    washer_duration: Distribution = Field(..., description="Distribution for washer cycle duration (hours)")
    dryer_duration: Distribution = Field(..., description="Distribution for dryer cycle duration (hours)")
    start_time_event_assumptions: EventAssumptions = Field(..., description="Event timing assumptions for laundry start times")



class RefrigeratorAssumptions(BaseModel):
    """Power assumptions for refrigeration equipment in Watts."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    efficient_small: Distribution = Field(..., description="Distribution for efficient small refrigerator power (W)")
    inefficient_small: Distribution = Field(..., description="Distribution for inefficient small refrigerator power (W)")  
    efficient_medium: Distribution = Field(..., description="Distribution for efficient medium refrigerator power (W)")
    inefficient_medium: Distribution = Field(..., description="Distribution for inefficient medium refrigerator power (W)")  
    efficient_large: Distribution = Field(..., description="Distribution for efficient large refrigerator power (W)")
    inefficient_large: Distribution = Field(..., description="Distribution for inefficient large refrigerator power (W)")

class DishwasherAssumptions(BaseModel):
    """Power assumptions for dishwasher equipment in Watts."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    efficient_dishwasher: Distribution = Field(..., description="Distribution for efficient dishwasher power (W)")
    inefficient_dishwasher: Distribution = Field(..., description="Distribution for inefficient dishwasher power (W)")
    dishwasher_cycle_duration: Distribution = Field(..., description="Distribution for dishwasher cycle duration (hours)")
    start_time_event_assumptions: EventAssumptions = Field(..., description="Event timing assumptions for dishwasher start times")

class CookingAssumptions(BaseModel):
    """Power assumptions for cooking products in Watts."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    electric_cooking_products: Distribution = Field(..., description="Distribution for efficient cooking products power (W)")
    weekday_breakfast_duration: Distribution = Field(..., description="Duration of cooking in hours for weekday breakfast (before 10am)")
    weekday_lunch_duration: Distribution = Field(..., description="Duration of cooking in hours for weekday lunch (10am - 3pm)")
    weekday_dinner_duration: Distribution = Field(..., description="Duration of cooking in hours for weekday dinner (after 3pm)")
    weekend_breakfast_duration: Distribution = Field(..., description="Duration of cooking in hours for weekend breakfast (before 10am)")
    weekend_lunch_duration: Distribution = Field(..., description="Duration of cooking in hours for weekend lunch (10am - 3pm)")
    weekend_dinner_duration: Distribution = Field(..., description="Duration of cooking in hours for weekend dinner (after 3pm)")
    start_time_event_assumptions: EventAssumptions = Field(..., description="Event timing assumptions for cooking start times")
    # TODO: add morning power fraction if needed
    # TODO: add gas cooking products if needed and potentially other fuel types


    

class EquipmentAssumptions(BaseModel):
    """Master configuration for equipment power assumptions."""
    baseload: float = Field(..., description="Base load in Watts")
    resolution_mins: int = Field(..., description="Time resolution in minutes (multiples of 5) for equipment usage schedules")
    laundry: LaundryAssumptions
    refrigerator: RefrigeratorAssumptions
    dishwasher: DishwasherAssumptions
    cooking_products: CookingAssumptions
   

    @classmethod
    def from_json_file(cls, path: str | Path) -> "EquipmentAssumptions":
        """Load equipment assumptions from a JSON file.
        JSON should contain csv_path fields for events, which will be loaded at the specified resolution."""
        data = json.loads(Path(path).read_text())
        resolution = data.get('resolution_mins', 15)

        # Load laundry
        if 'laundry' in data and 'start_time_event_csv_path' in data['laundry']:
            csv_path = data['laundry'].pop('start_time_event_csv_path')
            data['laundry']['start_time_event_assumptions'] = EventAssumptions.from_csv_file(csv_path, resolution)

        # Load dishwasher
        if 'dishwasher' in data and 'start_time_event_csv_path' in data['dishwasher']:
            csv_path = data['dishwasher'].pop('start_time_event_csv_path')
            data['dishwasher']['start_time_event_assumptions'] = EventAssumptions.from_csv_file(csv_path, resolution)

        # Load cooking
        if 'cooking_products' in data and 'start_time_event_csv_path' in data['cooking_products']:
            csv_path = data['cooking_products'].pop('start_time_event_csv_path')
            data['cooking_products']['start_time_event_assumptions'] = EventAssumptions.from_csv_file(csv_path, resolution)

        return cls(**data)

    @classmethod
    def default(cls) -> "EquipmentAssumptions":
        """Returns the standard/default assumptions for equipment power."""
        resolution_mins = 15
        return cls(
            # TODO: update the baseload to match
            baseload=1000.0,
            resolution_mins=resolution_mins,
            laundry=LaundryAssumptions(
                efficient_washer=DistributionConfig(dist_type="normal", params={"mean": 410.0, "std": 135.0, "lower": 0.0, "int": False}).build(),
                inefficient_washer=DistributionConfig(dist_type="normal", params={"mean": 700.0, "std": 155.0,  "lower": 0.0, "int": False}).build(),
                efficient_dryer=DistributionConfig(dist_type="normal", params={"mean": 1900.0, "std": 500.0,  "lower": 0.0,"int": False}).build(),
                inefficient_dryer=DistributionConfig(dist_type="normal", params={"mean": 3500.0, "std": 700.0, "lower": 0.0, "int": False}).build(),
                washer_duration=DistributionConfig(dist_type="normal", params={"mean": 1.25, "std": 0.25, "lower": 0.0, "int": False}).build(),
                dryer_duration=DistributionConfig(dist_type="normal", params={"mean": 1.0, "std": 0.25, "lower": 0.0, "int": False}).build(),
                start_time_event_assumptions=EventAssumptions.from_csv_file("data/equipment/laundry_start_time_5min_bins.csv", resolution_mins)
            ),
            refrigerator=RefrigeratorAssumptions(
                efficient_small=DistributionConfig(dist_type="normal", params={"mean": 32.0, "std": 6.5, "lower": 0.0, "int": False}).build(),
                inefficient_small=DistributionConfig(dist_type="normal", params={"mean": 51.0, "std": 7.5, "lower": 0.0, "int": False}).build(),
                efficient_medium=DistributionConfig(dist_type="normal", params={"mean": 48.0, "std": 11.0, "lower": 0.0, "int": False}).build(),
                inefficient_medium=DistributionConfig(dist_type="normal", params={"mean": 77.0, "std": 13.0, "lower": 0.0, "int": False}).build(),
                efficient_large=DistributionConfig(dist_type="normal", params={"mean": 69.0, "std": 13.0, "lower": 0.0, "int": False}).build(),
                inefficient_large=DistributionConfig(dist_type="normal", params={"mean": 110.0, "std": 17.5, "lower": 0.0, "int": False}).build(),
            ),
            dishwasher=DishwasherAssumptions(
                efficient_dishwasher=DistributionConfig(dist_type="normal", params={"mean": 540.0, "std": 50.0, "lower": 0.0, "int": False}).build(),
                inefficient_dishwasher=DistributionConfig(dist_type="normal", params={"mean": 810.0, "std": 75.0, "lower": 0.0, "int": False}).build(),
                dishwasher_cycle_duration=DistributionConfig(dist_type="normal", params={"mean": 2.0, "std": 0.75, "lower": 0.0, "int": False}).build(),
                start_time_event_assumptions=EventAssumptions.from_csv_file("data/equipment/dishwasher_start_time_5min_bins.csv", resolution_mins)
            ),
            cooking_products=CookingAssumptions(
                electric_cooking_products=DistributionConfig(dist_type="normal", params={"mean": 900.0, "std": 15.0, "lower": 0.0, "int": False}).build(),
                weekday_breakfast_duration=DistributionConfig(dist_type="normal", params={"mean": 0.3, "std": 0.25, "lower": 0.0, "int": False}).build(),
                weekday_lunch_duration=DistributionConfig(dist_type="normal", params={"mean": 0.5, "std": 0.4, "lower": 0.0, "int": False}).build(),
                weekday_dinner_duration=DistributionConfig(dist_type="normal", params={"mean": 0.6, "std": 0.4, "lower": 0.0, "int": False}).build(),
                weekend_breakfast_duration=DistributionConfig(dist_type="normal", params={"mean": 0.4, "std": 0.4, "lower": 0.0, "int": False}).build(),
                weekend_lunch_duration=DistributionConfig(dist_type="normal", params={"mean": 0.6, "std": 0.5, "lower": 0.0, "int": False}).build(),
                weekend_dinner_duration=DistributionConfig(dist_type="normal", params={"mean": 0.6, "std": 0.4, "lower": 0.0, "int": False}).build(),
                start_time_event_assumptions=EventAssumptions.from_csv_file("data/activity_initial_probability/cooking_start_time_5min_bins.csv", resolution_mins)
            )
        )


class EquipmentGenerator:

    def __init__(self, equipment: Equipment.Equipment, active_mask : list[list[bool]], resolution_mins : int, equipment_assumptions: EquipmentAssumptions = EquipmentAssumptions.default()):
        self.equipment = equipment
        self.active_time_mask = active_mask
        self.resolution_mins = resolution_mins
        self.equipment_assumptions = equipment_assumptions
       
        

    # def get_equipment_usage_schedule(self) -> dict[str, list[float]]:
    #     """ Translates equipment usage schedule into a full week schedule with power."""
    #     laundry_schedule = self.get_laundry_usage_schedule() if self.laundry_schedule is None else self.laundry_schedule
    #     fridge_schedule = self.get_fridge_usage_schedule() if self.fridge_schedule is None else self.fridge_schedule
    #     dishwasher_schedule = self.get_dishwasher_usage_schedule() if self.dishwasher_schedule is None else self.dishwasher_schedule
    #     electric_kitchen,gas_kitchen = self.get_kitchen_usage_schedule() if self.kitchen_electric_schedule is None or self.kitchen_gas_schedule is None else (self.kitchen_electric_schedule, self.kitchen_gas_schedule)
    #     base_load = [ EquipmentTranslator.system_base_load] * (24 * 7)
    #     sum = np.array(laundry_schedule) + np.array(fridge_schedule) + np.array(dishwasher_schedule) + np.array(electric_kitchen) + np.array(base_load)
    #     equipment_schedule = sum.tolist()
    #     return equipment_schedule
    
    # def _get_valid_start_times(self, duration: int) -> list[int]:
    #     valid_starts = []
    #     for start_hour in range(24 * 7):
    #         # Check if all hours from start to start+duration are active
    #         all_active = True
    #         for h in range(duration):
    #             hour_idx = (start_hour + h) % (24 * 7)
    #             if not self.active_time_mask[hour_idx]:
    #                 all_active = False
    #                 break
    #         if all_active:
    #             valid_starts.append(start_hour)

    #     if not valid_starts:
    #         return None
    #     return valid_starts

    def _get_laundry_power(self) -> tuple[float, float]:
        """ Calculates laundry equipment power density in W."""
        washer_power = 0.0
        dryer_power = 0.0
        laundry = self.equipment.laundry
        assumptions = self.equipment_assumptions.laundry
        if laundry.has_washer:
            if laundry.washer_efficient:
                washer_power = assumptions.efficient_washer.sample()
            else:
                washer_power = assumptions.inefficient_washer.sample()
        if laundry.has_dryer:
            if laundry.dryer_efficient:
                dryer_power = assumptions.efficient_dryer.sample()
            else:
                dryer_power = assumptions.inefficient_dryer.sample()
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
        refrigerator_assumptions = self.equipment_assumptions.refrigerator
        if refrigeration.has_refrigerator:
            if refrigeration.size == Equipment.RefrigerationSize.SMALL:
                base_power_dist = refrigerator_assumptions.efficient_small if refrigeration.efficient_refrigerator else refrigerator_assumptions.inefficient_small
            elif refrigeration.size == Equipment.RefrigerationSize.MEDIUM:
                base_power_dist = refrigerator_assumptions.efficient_medium if refrigeration.efficient_refrigerator else refrigerator_assumptions.inefficient_medium
            elif refrigeration.size == Equipment.RefrigerationSize.LARGE:
                base_power_dist = refrigerator_assumptions.efficient_large if refrigeration.efficient_refrigerator else refrigerator_assumptions.inefficient_large
            base_power = base_power_dist.sample()
            return base_power
        else:
            return 0.0
        
    def get_fridge_usage_schedule(self, weekly_active_mask:list[bool]) -> list[float]:
        """ Translates refrigeration equipment usage schedule into a full week schedule with power."""
        if self.equipment.refrigeration.has_refrigerator:
            fridge_power = self._get_fridge_power()
            fridge_schedule = [fridge_power for _ in range(len(weekly_active_mask))]
            return fridge_schedule
        else:
            fridge_schedule = [0.0 for _ in range(len(weekly_active_mask))]
            return fridge_schedule
        
    def _get_dishwasher_power (self) -> float:
        """ Calculates dishwasher equipment power density in W."""
        dishwasher = self.equipment.dishwasher
        assumptions = self.equipment_assumptions.dishwasher
        if dishwasher.has_dishwasher:
            if dishwasher.dishwasher_efficient:
                dishwasher_power = assumptions.efficient_dishwasher.sample()
            else:
                dishwasher_power = assumptions.inefficient_dishwasher.sample()
            return dishwasher_power
        else:
            return 0.0

    # def get_dishwasher_usage_schedule(self) -> list[float]:
    #     """ Translates dishwasher equipment usage schedule into a full week schedule with power."""
    #     if self.dishwasher_schedule is not None:
    #         return self.dishwasher_schedule
    #     dishwasher = self.equipment.dishwasher
    #     if dishwasher.has_dishwasher:
    #         dishwasher_power = self._get_dishwasher_power()
            
    #         num_cycle_dist = TranslationRule.RuleSet.uniform_distribution_rule()(dishwasher.usage_frequency_per_week.min, dishwasher.usage_frequency_per_week.max)
    #         num_cycles = int(num_cycle_dist.sample())
    #         cycle_duration = EquipmentTranslator.dishwasher_cycle_duration_dist.sample()
            
    #         valid_start_times = self._get_valid_start_times(cycle_duration)
    #         if valid_start_times is None:
    #             valid_start_times = self._get_valid_start_times(0)

    #         dishwasher_schedule = [0.0] * (24 * 7)
            
    #         if len(valid_start_times) == 0:
    #             self.dishwasher_schedule = dishwasher_schedule
    #             return self.dishwasher_schedule
    #         else:
    #             # more weights for evening times, moderate for after lunch, less for the rest, valid times only
    #             # more weights for friday evening and weekends
    #             weighted_prob = {
    #                 3.0: [],
    #                 2.0: [],
    #                 1.5: [],
    #                 1.0: []
    #             }
    #             for s in valid_start_times:
    #                 day_of_week, hour_of_day = Utils.hour_of_day(s)
    #                 if day_of_week in [5, 6]:  # Saturday, Sunday
    #                     if hour_of_day >= 18:
    #                         weighted_prob[3.0].append(s)
    #                     elif 12 <= hour_of_day < 16:
    #                         weighted_prob[2.0].append(s)
    #                     else:
    #                         weighted_prob[1.5].append(s)
    #                 if day_of_week == 4:  # Friday
    #                     if hour_of_day >= 18:
    #                         weighted_prob[3.0].append(s)
    #                     elif 12 <= hour_of_day < 16:
    #                         weighted_prob[1.5].append(s)
    #                     else:
    #                         weighted_prob[1.0].append(s)
    #                 else:  # Weekdays
    #                     if hour_of_day >= 19:
    #                         weighted_prob[2.0].append(s)
    #                     elif 12 <= hour_of_day < 14:
    #                         weighted_prob[1.5].append(s)
    #                     else:
    #                         weighted_prob[1.0].append(s)
    #             start_time_dist = TranslationRule.RuleSet.weighted_value_distribution_rule()(weighted_prob)
    #             for _ in range(num_cycles):
    #                 start_time = start_time_dist.sample()
    #                 if start_time is None:
    #                     self.dishwasher_schedule = dishwasher_schedule
    #                     return self.dishwasher_schedule
    #                 for h in range(cycle_duration):
    #                     hour_idx = (start_time + h) % (24 * 7)
    #                     dishwasher_schedule[hour_idx] = dishwasher_power
    #                 # update weighted prob to avoid overlapping cycles
    #                 update_weighted_prob = { 
    #                     0.0: [s for s in valid_start_times if start_time <= s < start_time + cycle_duration],
    #                     0.2: [s for s in valid_start_times if not (start_time <= s < start_time + cycle_duration) and (abs(s - start_time) < 2 or abs(s - start_time - cycle_duration) <2) ],
    #                 }

    #                 start_time_dist.update_weights_by_factor(update_weighted_prob)
    #             self.dishwasher_schedule = dishwasher_schedule
    #             return self.dishwasher_schedule
    #     else:
    #         self.dishwasher_schedule = [0.0] * (24 * 7)
    #         return self.dishwasher_schedule
        
    def _get_cooking_dist(self) -> float:
        cooking_products = self.equipment.cooking_products
        assumptions = self.equipment_assumptions.cooking_products
        if cooking_products.has_cooking_products:
            if cooking_products.cooktop_fuel == Equipment.FuelType.ELECTRIC:
                return assumptions.electric_cooking_products.sample()
            else:
                raise NotImplementedError("non-electric cooking products to be implemented.")
        else:
            return 0.0
        

    
    # def get_kitchen_usage_schedule(self) -> tuple[list[float], list[float]]:
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