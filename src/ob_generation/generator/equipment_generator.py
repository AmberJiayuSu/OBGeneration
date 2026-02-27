from ob_generation.stochastic.distribution import Distribution, UniformDistribution, CategoricalDistribution
from ob_generation.generator.ob_utils import ScheduleUtils, get_project_root
import ob_generation.model.equipment as Equipment
from pydantic import BaseModel, Field, ConfigDict
from ob_generation.stochastic.distribution_config import DistributionConfig
import json
from pathlib import Path
import csv


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

    def get_masked_probabilities(self, active_mask: list[bool], match_length=False) -> list[float]:
        """Apply active occupancy mask and return renormalized probabilities.
        Does not mutate the original probabilities.

        Args:
            active_mask: Boolean mask indicating active time bins (same length as probabilities)
        Returns:
            List of masked and renormalized probabilities
        """
        if not match_length:
            if len(active_mask) != len(self.start_time_probabilities):
                raise ValueError(f"Active mask length {len(active_mask)} does not match probabilities length {len(self.start_time_probabilities)}")
        
            masked_probs = [prob if active_mask[i] else 0.0 for i, prob in enumerate(self.start_time_probabilities)]
        else:
            masked_probs = [self.start_time_probabilities[i] if active_mask[i] else 0.0 for i in range(len(active_mask))]

        total = sum(masked_probs)
        if total > 0:
            return [p / total for p in masked_probs]
        else:
            # Masking makes all probabilities zero
            num_active = sum(active_mask)
            if num_active > 0:
                # Assume sometime occupied, then assume uniform distribution over active times
                return [1.0 / num_active if active_mask[i] else 0.0 for i in range(len(active_mask))]
            else:
                # If never occupied, return all zeros
                return masked_probs  
    


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
    weekday_same_day_downfactor: float = Field(..., description="Downfactor for same day weekday laundry events probabilities")   
    weekend_same_day_downfactor: float = Field(..., description="Downfactor for same day weekend laundry events probabilities")



class RefrigeratorAssumptions(BaseModel):
    """Power assumptions for refrigeration equipment in Watts."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    efficient_compact: Distribution = Field(..., description="Distribution for efficient compact refrigerator power (W)")
    inefficient_compact: Distribution = Field(..., description="Distribution for inefficient compact refrigerator power (W)")
    efficient_small: Distribution = Field(..., description="Distribution for efficient small refrigerator power (W)")
    inefficient_small: Distribution = Field(..., description="Distribution for inefficient small refrigerator power (W)")  
    efficient_medium: Distribution = Field(..., description="Distribution for efficient medium refrigerator power (W)")
    inefficient_medium: Distribution = Field(..., description="Distribution for inefficient medium refrigerator power (W)")  
    efficient_large: Distribution = Field(..., description="Distribution for efficient large refrigerator power (W)")
    inefficient_large: Distribution = Field(..., description="Distribution for inefficient large refrigerator power (W)")
    primary_refrigerator_size: CategoricalDistribution = Field(..., description="Categorical distribution for primary refrigerator size")
    secondary_refrigerator_size: CategoricalDistribution = Field(..., description="Categorical distribution for secondary refrigerator size")

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
    watts_per_person_active : float = Field(..., description="Additional watts per active person in Watts")
    watts_per_person_sleep: float = Field(..., description="Additional watts per sleeping person in Watts")
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
    def default(cls, resolution_mins: int = 15) -> "EquipmentAssumptions":
        """Returns the standard/default assumptions for equipment power."""
        # Get project root directory using robust path resolution
        project_root = get_project_root()
        data_dir = project_root / "data" / "activity_initial_probability"
        return cls(
            baseload=250.0,
            watts_per_person_active=120.0,
            watts_per_person_sleep=40.0,
            resolution_mins=resolution_mins,
            laundry=LaundryAssumptions(
                efficient_washer=DistributionConfig(dist_type="normal", params={"mean": 410.0, "std": 135.0, "lower": 0.0, "int": False}).build(),
                inefficient_washer=DistributionConfig(dist_type="normal", params={"mean": 700.0, "std": 155.0,  "lower": 0.0, "int": False}).build(),
                efficient_dryer=DistributionConfig(dist_type="normal", params={"mean": 1900.0, "std": 500.0,  "lower": 0.0,"int": False}).build(),
                inefficient_dryer=DistributionConfig(dist_type="normal", params={"mean": 3500.0, "std": 700.0, "lower": 0.0, "int": False}).build(),
                washer_duration=DistributionConfig(dist_type="normal", params={"mean": 1.25, "std": 0.25, "lower": 0.0, "int": False}).build(),
                dryer_duration=DistributionConfig(dist_type="normal", params={"mean": 1.0, "std": 0.25, "lower": 0.0, "int": False}).build(),
                start_time_event_assumptions=EventAssumptions.from_csv_file(str(data_dir / "laundry_start_time_5min_bins.csv"), resolution_mins),
                weekday_same_day_downfactor=0.22,
                weekend_same_day_downfactor=0.27
            ),
            refrigerator=RefrigeratorAssumptions(
                efficient_compact=DistributionConfig(dist_type="normal", params={"mean": 31.0, "std": 6.0, "lower": 0.0, "int": False}).build(),
                inefficient_compact=DistributionConfig(dist_type="normal", params={"mean": 48.0, "std": 11.0, "lower": 0.0, "int": False}).build(),
                efficient_small=DistributionConfig(dist_type="normal", params={"mean": 48.0, "std": 11.0, "lower": 0.0, "int": False}).build(),
                inefficient_small=DistributionConfig(dist_type="normal", params={"mean": 72.0, "std": 13.0, "lower": 0.0, "int": False}).build(),
                efficient_medium=DistributionConfig(dist_type="normal", params={"mean": 61.0, "std": 12.0, "lower": 0.0, "int": False}).build(),
                inefficient_medium=DistributionConfig(dist_type="normal", params={"mean": 91.5, "std": 18.0, "lower": 0.0, "int": False}).build(),
                efficient_large=DistributionConfig(dist_type="normal", params={"mean": 69.0, "std": 13.0, "lower": 0.0, "int": False}).build(),
                inefficient_large=DistributionConfig(dist_type="normal", params={"mean": 103.5, "std": 21.0, "lower": 0.0, "int": False}).build(),
                primary_refrigerator_size=CategoricalDistribution(probabilities=[0.0072, 0.0506, 0.5013, 0.4409]), # compact, small, medium, large
                secondary_refrigerator_size=CategoricalDistribution(probabilities=[0.2570, 0.1790, 0.4079, 0.1561])

            ),
            dishwasher=DishwasherAssumptions(
                efficient_dishwasher=DistributionConfig(dist_type="normal", params={"mean": 540.0, "std": 50.0, "lower": 0.0, "int": False}).build(),
                inefficient_dishwasher=DistributionConfig(dist_type="normal", params={"mean": 810.0, "std": 75.0, "lower": 0.0, "int": False}).build(),
                dishwasher_cycle_duration=DistributionConfig(dist_type="normal", params={"mean": 2.0, "std": 0.75, "lower": 0.0, "int": False}).build(),
                start_time_event_assumptions=EventAssumptions.from_csv_file(str(data_dir / "dishwasher_start_time_5min_bins.csv"), resolution_mins)
            ),
            cooking_products=CookingAssumptions(
                electric_cooking_products=DistributionConfig(dist_type="normal", params={"mean": 900.0, "std": 15.0, "lower": 0.0, "int": False}).build(),
                weekday_breakfast_duration=DistributionConfig(dist_type="normal", params={"mean": 0.3, "std": 0.25, "lower": 0.0, "int": False}).build(),
                weekday_lunch_duration=DistributionConfig(dist_type="normal", params={"mean": 0.5, "std": 0.4, "lower": 0.0, "int": False}).build(),
                weekday_dinner_duration=DistributionConfig(dist_type="normal", params={"mean": 0.6, "std": 0.4, "lower": 0.0, "int": False}).build(),
                weekend_breakfast_duration=DistributionConfig(dist_type="normal", params={"mean": 0.4, "std": 0.4, "lower": 0.0, "int": False}).build(),
                weekend_lunch_duration=DistributionConfig(dist_type="normal", params={"mean": 0.6, "std": 0.5, "lower": 0.0, "int": False}).build(),
                weekend_dinner_duration=DistributionConfig(dist_type="normal", params={"mean": 0.6, "std": 0.4, "lower": 0.0, "int": False}).build(),
                start_time_event_assumptions=EventAssumptions.from_csv_file(str(data_dir / "cooking_start_time_5min_bins.csv"), resolution_mins)
            )
        )


class EquipmentGenerator:

    def __init__(self, equipment: Equipment.Equipment, active_mask : list[list[bool]], sleep_mask: list[list[bool]], occupancy: list[list[float]],num_occupants: int, resolution_mins : int, equipment_assumptions: EquipmentAssumptions = EquipmentAssumptions.default()):
        self.equipment = equipment
        self.active_time_mask = active_mask
        self.sleep_time_mask = sleep_mask
        self.occupancy = occupancy
        self.num_occupants = num_occupants
        self.resolution_mins = resolution_mins
        self.equipment_assumptions = equipment_assumptions

        if self.equipment.laundry.usage_frequency_per_week is not None:
            self.laundry_frequency_dist = UniformDistribution(
                lower=self.equipment.laundry.usage_frequency_per_week.min,
                upper=self.equipment.laundry.usage_frequency_per_week.max,
                int=True
            )
        if self.equipment.cooking_products.usage_frequency_per_week is not None:
            self.cooking_frequency_dist = UniformDistribution(
                lower=self.equipment.cooking_products.usage_frequency_per_week.min,
                upper=self.equipment.cooking_products.usage_frequency_per_week.max,
                int=True
            )
        if self.equipment.dishwasher.dishwashing_operational_logic.usage_frequency_per_week is not None:
            self.dishwasher_frequency_dist = UniformDistribution(
                lower=self.equipment.dishwasher.dishwashing_operational_logic.usage_frequency_per_week.min,
                upper=self.equipment.dishwasher.dishwashing_operational_logic.usage_frequency_per_week.max,
                int=True
            )

    def equipment_annual_schedule(self) -> tuple[list[list[float]], list[list[int]], list[list[int]]]:
        """ Translates equipment usage pattern into a full annual schedule based on occupancy and sleep times.
            Returns: Tuple of (annual_equipment_schedule, laundry_cycles, dishwasher_cycles)"""
        annual_schedule = []
        laundry_cycles = []
        dishwasher_cycles = []
        for week_index in range(len(self.active_time_mask)):
            weekly_active_mask = self.active_time_mask[week_index]
            weekly_sleep_mask = self.sleep_time_mask[week_index]
            weekly_occupancy = self.occupancy[week_index]
            last_week = (week_index == len(self.active_time_mask) -1)
            weekly_equipment_schedule, laundry_num_cycles, dishwasher_num_cycles = self.equipment_weekly_schedule(weekly_active_mask, weekly_sleep_mask, weekly_occupancy, last_week)
            annual_schedule.append(weekly_equipment_schedule)
            laundry_cycles.append(laundry_num_cycles)
            dishwasher_cycles.append(dishwasher_num_cycles)
        return annual_schedule, laundry_cycles, dishwasher_cycles

    def equipment_weekly_schedule(self, weekly_active_mask: list[bool], weekly_sleep_mask: list[bool], weekly_occupancy: list[float], last_week:bool) -> tuple[list[float], int, int]:
        """ Translates equipment usage pattern into a full week schedule based on occupancy and sleep times."""
        baseload_schedule = self.weekly_baseload_schedule(weekly_active_mask, weekly_sleep_mask, weekly_occupancy)
        laundry_schedule, laundry_num_cycles = self.weekly_laundry_usage_schedule(weekly_active_mask, last_week)
        fridge_power = self._get_fridge_power()
        cooking_schedule, cooking_ends = self.weekly_cooking_usage_schedule(weekly_active_mask, last_week)
        dishwashing_schedule, dishwasher_num_cycles = self.weekly_dishwasher_usage_schedule(weekly_active_mask,cooking_ends,last_week)

        total_schedule = []
        for i in range(len(weekly_active_mask)):
            total_power = baseload_schedule[i] + laundry_schedule[i] + fridge_power + cooking_schedule[i] + dishwashing_schedule[i]
            total_schedule.append(total_power)
        return total_schedule, laundry_num_cycles, dishwasher_num_cycles 


    def weekly_baseload_schedule(self, weekly_active_mask: list[bool], weekly_sleep_mask: list[bool], weekly_occupancy: list[float]) -> list[float]:
        assumptions = self.equipment_assumptions
        baseload_schedule = [assumptions.baseload] * len(weekly_active_mask)
        for i in range(len(weekly_active_mask)):
            if weekly_active_mask[i]:
                baseload_schedule[i] += weekly_occupancy[i] * assumptions.watts_per_person_active * self.num_occupants
            elif weekly_sleep_mask[i]:
                baseload_schedule[i] += weekly_occupancy[i] * assumptions.watts_per_person_sleep * self.num_occupants
        return baseload_schedule
    

    def laundry_annual_schedule(self) ->list[list[float]]:
        """ Translates laundry equipment usage pattern into a full annual schedule based on occupancy."""
        annual_schedule = []
        laundry_cycles = []
        for week_index in range(len(self.active_time_mask)):
            weekly_active_mask = self.active_time_mask[week_index]
            last_week = (week_index == len(self.active_time_mask) -1)
            weekly_laundry_schedule, weekly_num_cycles = self.weekly_laundry_usage_schedule(weekly_active_mask, last_week)
            annual_schedule.append(weekly_laundry_schedule)
            laundry_cycles.append(weekly_num_cycles)
        return annual_schedule
    
    def cooking_annual_schedule(self) -> tuple[list[list[float]], list[list[float]]]:
        """ Translates cooking equipment usage pattern into a full annual schedule based on occupancy."""
        annual_schedule = []
        annual_cooking_ends = []
        for week_index in range(len(self.active_time_mask)):
            weekly_active_mask = self.active_time_mask[week_index]
            last_week = (week_index == len(self.active_time_mask) -1)
            weekly_cooking_schedule, cooking_ends = self.weekly_cooking_usage_schedule(weekly_active_mask, last_week)
            annual_schedule.append(weekly_cooking_schedule)
            annual_cooking_ends.append(cooking_ends)
        return annual_schedule, annual_cooking_ends
    
    def dishwasher_annual_schedule(self, annual_cooking_ends) ->list[list[float]]:
        """ Translates dishwasher equipment usage pattern into a full annual schedule based on occupancy."""
        annual_schedule = []
        dishwasher_cycles = []
        for week_index in range(len(self.active_time_mask)):
            weekly_active_mask = self.active_time_mask[week_index]
            last_week = (week_index == len(self.active_time_mask) -1)
            cooking_ends = annual_cooking_ends[week_index]
            weekly_dishwashing_schedule, weekly_num_cycles = self.weekly_dishwasher_usage_schedule(weekly_active_mask,cooking_ends, last_week)
            annual_schedule.append(weekly_dishwashing_schedule)
            dishwasher_cycles.append(weekly_num_cycles)
        return annual_schedule



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
    
    def weekly_laundry_usage_schedule(self, weekly_active_mask: list[bool], last_week: bool) -> tuple[list[float], list[int]]:
        """ Translates laundry equipment usage schedule into a full week schedule with power."""
        laundry = self.equipment.laundry
        laundry_assumptions = self.equipment_assumptions.laundry
        res_min = self.resolution_mins
        start_time_prob = self.equipment_assumptions.laundry.start_time_event_assumptions.get_masked_probabilities(weekly_active_mask,last_week)
        start_time_dist = CategoricalDistribution(start_time_prob)
        total_num_days = len(weekly_active_mask) // (24 * (60 // res_min))
        if laundry.has_washer or laundry.has_dryer:
            weekly_num_cycles = [0] * total_num_days
            washer_power, dryer_power = self._get_laundry_power() # W
            num_cycle = self.laundry_frequency_dist.sample()
            laundry_schedule = [0.0] * (len(weekly_active_mask))
            cnt = 0
            while cnt < num_cycle:
                start_index = start_time_dist.sample()
                washer_duration = laundry_assumptions.washer_duration.sample() if laundry.has_washer else 0.0
                dryer_duration = laundry_assumptions.dryer_duration.sample() if laundry.has_dryer else 0.0
                washer_end = start_index + round((washer_duration*60) / res_min)
                dryer_end = washer_end + round((dryer_duration*60) / res_min)
                if dryer_end > len(weekly_active_mask):
                    continue  # skip this cycle if it exceeds the week
                cnt += 1
                for i in range(start_index, washer_end):
                    laundry_schedule[i] = washer_power
                for i in range(washer_end, dryer_end):
                    laundry_schedule[i] = dryer_power

                day = ScheduleUtils.weekly_index_day(start_index, res_min)
                weekly_num_cycles[day] += 1
                downfactor = laundry_assumptions.weekday_same_day_downfactor if day < 5 else laundry_assumptions.weekend_same_day_downfactor
                prob_factor_update = {
                    0.0: [i for i in range(start_index, dryer_end)],
                    downfactor : ScheduleUtils.get_day_indices(day, res_min)
                }
                start_time_dist.update_probabilities_by_factor(prob_factor_update)
            return laundry_schedule , weekly_num_cycles
        else:
            return [0.0] * (len(weekly_active_mask)), [0] * total_num_days
        


    def _get_fridge_power(self) -> float:
        """ Calculates refrigeration equipment power density in W."""
        refrigerator = self.equipment.refrigerator
        refrigerator_assumptions = self.equipment_assumptions.refrigerator
        total_power = 0.0
        if refrigerator.has_refrigerator:
            for i in range(refrigerator.number_of_refrigerators):
                if i == 0:
                    size = refrigerator_assumptions.primary_refrigerator_size.sample()
                else:
                    size = refrigerator_assumptions.secondary_refrigerator_size.sample()
                if size == 0:
                    power_dist = refrigerator_assumptions.efficient_compact if refrigerator.efficient_refrigerator else refrigerator_assumptions.inefficient_compact
                elif size == 1:
                    power_dist = refrigerator_assumptions.efficient_small if refrigerator.efficient_refrigerator else refrigerator_assumptions.inefficient_small
                elif size == 2:
                    power_dist = refrigerator_assumptions.efficient_medium if refrigerator.efficient_refrigerator else refrigerator_assumptions.inefficient_medium
                else:
                    power_dist = refrigerator_assumptions.efficient_large if refrigerator.efficient_refrigerator else refrigerator_assumptions.inefficient_large
                power = power_dist.sample()
                total_power += power
            return total_power
        else:
            return 0.0
        
      
    def _get_cooking_power(self) -> float:
        cooking_products = self.equipment.cooking_products
        assumptions = self.equipment_assumptions.cooking_products
        if cooking_products.has_cooking_products:
            if cooking_products.cooking_products_fuel == Equipment.FuelType.ELECTRIC:
                return assumptions.electric_cooking_products.sample()
            else:
                raise NotImplementedError("non-electric cooking products to be implemented.")
        else:
            return 0.0
        

    def weekly_cooking_usage_schedule(self, weekly_active_mask:list[bool], last_week: bool) -> tuple[list[float], list[float]]:
        """ Translates kitchen equipment usage schedule into a full week schedule with power.
            Returns the cooking power schedule and the corresponding ending times."""
        cooking_products = self.equipment.cooking_products
        cooking_assumptions = self.equipment_assumptions.cooking_products
        masked_start_time_prob = cooking_assumptions.start_time_event_assumptions.get_masked_probabilities(weekly_active_mask,last_week)
        start_time_dist = CategoricalDistribution(masked_start_time_prob)
        cooking_schedule = [0.0] * (len(weekly_active_mask))
        num_per_hour = int(60 / self.resolution_mins)
        if not cooking_products.has_cooking_products:
            return cooking_schedule, []
        else:
            end_times = []
            num_cooking = self.cooking_frequency_dist.sample()
            cnt = 0
            while cnt < num_cooking:
                cooking_power = self._get_cooking_power()
                start_index = start_time_dist.sample()
                start_hour = ScheduleUtils.weekly_index_hour(start_index, self.resolution_mins)
                day = ScheduleUtils.weekly_index_day(start_index, self.resolution_mins)
                # Determine duration based on time of day and day of week
                if day < 5:  # Weekday
                    if start_hour < 10:
                        period_start = 0
                        perid_end = 10 * num_per_hour - 1
                        duration = cooking_assumptions.weekday_breakfast_duration.sample()
                    elif 10 <= start_hour < 15:
                        period_start = 10 * num_per_hour
                        perid_end = 15 * num_per_hour - 1
                        duration = cooking_assumptions.weekday_lunch_duration.sample()
                    else:
                        period_start = 15 * num_per_hour
                        perid_end = 24 * num_per_hour - 1
                        duration = cooking_assumptions.weekday_dinner_duration.sample()
                else:  # Weekend
                    if start_hour < 10:
                        period_start = 0
                        perid_end = 10 * num_per_hour - 1
                        duration = cooking_assumptions.weekend_breakfast_duration.sample()
                    elif 10 <= start_hour < 15: 
                        period_start = 10 * num_per_hour
                        perid_end = 15 * num_per_hour - 1
                        duration = cooking_assumptions.weekend_lunch_duration.sample()
                    else:
                        period_start = 15 * num_per_hour
                        perid_end = 24 * num_per_hour - 1
                        duration = cooking_assumptions.weekend_dinner_duration.sample()
                duration_indices = round(duration * 60 / self.resolution_mins)
                duration_indices = max(1, duration_indices)  # Ensure at least 1 index
                if start_index + duration_indices >= len(weekly_active_mask):
                    continue  # skip if exceeds week
                cnt += 1
                for i in range(start_index, start_index + duration_indices):
                    cooking_schedule[i] = cooking_power
                end_times.append((start_index + duration_indices) % (24 * 7 * num_per_hour))
                # Update probabilities to avoid overlapping cooking events
                update_prob_factor = {
                    0.0: [i for i in range(period_start, perid_end + 1) if start_index <= i < start_index + duration_indices]
                }
                start_time_dist.update_probabilities_by_factor(update_prob_factor)
            end_times.sort()
            return cooking_schedule, end_times

    def weekly_fridge_usage_schedule(self, weekly_active_mask:list[bool]) -> list[float]:
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

    def weekly_dishwasher_usage_schedule(self, weekly_active_mask:list[bool], cooking_ending:list[float], last_week: bool) -> tuple[list[float], list[int]]:
        """ Translates dishwasher equipment usage schedule into a full week schedule with power."""
        import time
        dishwasher = self.equipment.dishwasher
        dishwasher_assumptions = self.equipment_assumptions.dishwasher
        res_min = self.resolution_mins
        masked_time_prob = dishwasher_assumptions.start_time_event_assumptions.get_masked_probabilities(weekly_active_mask,last_week)
        start_time_dist = CategoricalDistribution(masked_time_prob)
        dishwasher_schedule = [0.0] * (len(weekly_active_mask))
        operation = dishwasher.dishwashing_operational_logic
        total_num_days = len(weekly_active_mask) // (24 * (60 // res_min))
        if not dishwasher.has_dishwasher:
            return dishwasher_schedule, [0] * total_num_days
        else:
            weekly_num_cycles = [0] * total_num_days
            if operation.pattern_type == Equipment.DishwashingPattern.AFTER_EACH_COOKED_MEAL:
                # Dishwasher runs after cooking events
                for end_time in cooking_ending:
                    start_index = end_time + 1
                    if start_index >= len(weekly_active_mask):
                        continue  # skip if exceeds week
                    day = ScheduleUtils.weekly_index_day(start_index, res_min) // 7
                    weekly_num_cycles[day] += 1
                    cycle_duration = round(dishwasher_assumptions.dishwasher_cycle_duration.sample() * 60 / res_min)
                    cycle_duration = max(1, cycle_duration)  # Ensure at least 1 index
                    cycle_duration = min (cycle_duration, len(weekly_active_mask) - start_index)  # Ensure does not exceed week
                    dishwasher_power = self._get_dishwasher_power()
                    for i in range(start_index, start_index + cycle_duration):
                        dishwasher_schedule[i] = dishwasher_power
            elif operation.pattern_type == Equipment.DishwashingPattern.DAILY_BATCH_IF_COOKED:
                days = set()
                for end_time in cooking_ending:
                    day = ScheduleUtils.weekly_index_day(end_time, res_min)
                    days.add(day)
                for day_idx, day in enumerate(days):
                    weekly_num_cycles[day] += 1
                    day_start = day * 24 * (60 // res_min)
                    day_end = day_start + 24 * (60 // res_min)
                    while True:
                        start_index = start_time_dist.sample_from_range(day_start, day_end)
                        cycle_duration = round(dishwasher_assumptions.dishwasher_cycle_duration.sample() * 60 / res_min)
                        cycle_duration = max(1, cycle_duration)  # Ensure at least 1 index
                        if start_index + cycle_duration >= len(weekly_active_mask):
                            continue  # skip if exceeds week
                        break
                    dishwasher_power = self._get_dishwasher_power()
                    for i in range(start_index, start_index + cycle_duration):
                        dishwasher_schedule[i] = dishwasher_power
            elif operation.pattern_type == Equipment.DishwashingPattern.WHENEVER_FULL:
                # assume dishwasher gets full after every 3 cooking events
                for i in range(0, len(cooking_ending), 3):
                    end_time = cooking_ending[i]
                    if (end_time + 1) >= len(weekly_active_mask):
                        continue  # skip if exceeds week
                    start_index = end_time + 1
                    day = ScheduleUtils.weekly_index_day(start_index, res_min)
                    weekly_num_cycles[day] += 1
                    day_end = (day + 1) * 24 * (60 // res_min)
                    # end_ind is the end index for the potential dishwasher cycle based on the next cooking event or the end of the day
                    end_ind = min(cooking_ending[i + 3], day_end) if i + 3 < len(cooking_ending) else day_end
                    attempt = 0
                    while attempt < 100:
                        attempt += 1
                        start_index = start_time_dist.sample_from_range(start_index, end_ind)
                        cycle_duration = round(dishwasher_assumptions.dishwasher_cycle_duration.sample() * 60 / res_min)
                        cycle_duration = max(1, cycle_duration)  # Ensure at least 1 index
                        actual_end_ind = start_index + cycle_duration
                        actual_end_ind = min(actual_end_ind, len(weekly_active_mask))
                        if actual_end_ind >= len(weekly_active_mask):
                            continue  # skip if exceeds week
                        break
                    dishwasher_power = self._get_dishwasher_power()
                    for i in range(start_index, actual_end_ind):
                        dishwasher_schedule[i] = dishwasher_power
            elif operation.pattern_type == Equipment.DishwashingPattern.INDEPENDENT_FREQUENCY:
                cnt = 0
                frequency_per_week = self.dishwasher_frequency_dist.sample()
                while cnt < frequency_per_week:
                    start_index = start_time_dist.sample()
                    cycle_duration = round(dishwasher_assumptions.dishwasher_cycle_duration.sample() * 60 / res_min)
                    cycle_duration = max(1, cycle_duration)  # Ensure at least 1 index
                    if start_index + cycle_duration >= len(weekly_active_mask):
                        continue  # skip if exceeds week
                    cnt += 1
                    day = ScheduleUtils.weekly_index_day(start_index, res_min)
                    weekly_num_cycles[day] += 1
                    dishwasher_power = self._get_dishwasher_power()
                    for i in range(start_index, start_index + cycle_duration):
                        dishwasher_schedule[i] = dishwasher_power
                    update_prob_factor = {
                        0.0: [i for i in range(start_index, start_index + cycle_duration)]
                    }
                    start_time_dist.update_probabilities_by_factor(update_prob_factor)
            return dishwasher_schedule, weekly_num_cycles
                
                    

         