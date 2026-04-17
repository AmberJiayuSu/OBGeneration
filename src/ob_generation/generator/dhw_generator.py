from pydantic import BaseModel, Field, ConfigDict
import json
from pathlib import Path


class DHWAssumptions(BaseModel):
    """DHW usage level assumptions."""
    model_config = ConfigDict(validate_assignment=True)
    hot_water_per_person_per_day: float = Field(..., description="Hot water usage per person per day in liters.")
    efficient_washer_per_cycle: float = Field(..., description="Hot water usage per washing machine cycle in liters.")
    inefficient_washer_per_cycle: float = Field(..., description="Hot water usage per washing machine cycle in liters.")
    efficient_dishwasher_per_cycle: float = Field( ... , description="Hot water usage per dishwasher cycle in liters.")
    inefficient_dishwasher_per_cycle: float = Field( ... , description="Hot water usage per dishwasher cycle in liters.")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DHWAssumptions":
        """Load DHW assumptions from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)
    
    @classmethod
    def default(cls) -> "DHWAssumptions":
        """Returns the standard/default assumptions for DHW usage.
           TODO: These values are deterministic, and we can rethink if we want to add some variability to them."""
        return cls(
            hot_water_per_person_per_day= 56.7812, # 15 gallons in liters
            efficient_washer_per_cycle=6.0,
            inefficient_washer_per_cycle=15.0,
            efficient_dishwasher_per_cycle=15.0,
            inefficient_dishwasher_per_cycle=30.0
        )
    
class DHWGenerator:
    
    def __init__(self, dhw_assumptions: DHWAssumptions, num_occupants: int, laundry_cycles_per_day: list[list[int]], dishwasher_cycles_per_day: list[list[int]], resolution_mins: int = 15):
        self.dhw_assumptions = dhw_assumptions
        #self.equipment = equipment
        self.num_occupants = num_occupants
        self.laundry_cycles_per_day = laundry_cycles_per_day
        self.dishwasher_cycles_per_day = dishwasher_cycles_per_day
        self.resolution_mins = resolution_mins

    @staticmethod
    def generate_with_defaults(
        num_occupants: int,
        laundry_cycles_per_day: list[list[int]],
        dishwasher_cycles_per_day: list[list[int]],
        resolution_mins: int
    ) -> tuple[float, list[list[float]]]:
        """Generates an annual DHW usage schedule in cubic meters per second using default assumptions."""
        generator = DHWGenerator(
            dhw_assumptions=DHWAssumptions.default(),
            num_occupants=num_occupants,
            laundry_cycles_per_day=laundry_cycles_per_day,
            dishwasher_cycles_per_day=dishwasher_cycles_per_day,
            resolution_mins=resolution_mins
        )
        return generator.dhw_annual_schedule()
       
    

    def dhw_annual_schedule(self) -> tuple[float, list[list[float]]]:
        """Generates an annual DHW usage schedule in cubic meters per second.
           Returns: Tuple of (max_flow_rate_m3_per_s, annual_dhw_schedule)"""
        annual_schedule = []
        annual_daily = []
        for week_index in range(len(self.laundry_cycles_per_day)):
            weekly_daily_dhw = self.dhw_weekly_schedule(self.laundry_cycles_per_day[week_index], self.dishwasher_cycles_per_day[week_index])
            annual_daily.append(weekly_daily_dhw) 
        max_flow = max(max(week) for week in annual_daily)  
        fraction = [ [day / max_flow for day in week] for week in annual_daily]
        flow_rate = self.litter_per_day_to_m3_per_second(max_flow)
        for week_fraction in fraction:
            week_schedule = []
            for day_fraction in week_fraction:
                week_schedule.extend([day_fraction] * (24 * (60 // self.resolution_mins)))
            annual_schedule.append(week_schedule)
        return flow_rate, annual_schedule



    def dhw_weekly_schedule(self, weekly_laundry_cycles_per_day: list[int], weekly_dishwasher_cycles_per_day: list[int]) -> list[float]:
        """Generates a weekly DHW usage schedule in cubic meters per second."""
        daily_occupants_dhw = self.occupants_dhw()
        daily_laundry_dhw = self.weekly_laundry_dhw(weekly_laundry_cycles_per_day)
        daily_dishwasher_dhw = self.weekly_dishwasher_dhw(weekly_dishwasher_cycles_per_day)
        total_daily_dhw = [daily_occupants_dhw + laundry + dishwasher for laundry, dishwasher in zip(daily_laundry_dhw, daily_dishwasher_dhw)]
        return total_daily_dhw


    def litter_per_day_to_m3_per_second(self, liters_per_day: float) -> float:
        """Converts liters per day to cubic meters per second."""
        cubic_meters_per_day = liters_per_day / 1000.0
        seconds_per_day = 24 * 3600
        return cubic_meters_per_day / seconds_per_day

    def occupants_dhw(self) -> float:
        """Calculates daily DHW usage based on number of occupants."""
        return self.num_occupants * self.dhw_assumptions.hot_water_per_person_per_day
    
    def weekly_laundry_dhw(self,  weekly_laundry_cycles_per_day:list[int]) -> float:
        """Calculates weekly DHW usage based on laundry cycles."""
        # laundry = self.equipment.laundry
        laundry = sum(weekly_laundry_cycles_per_day) # If there are no cycles, we can assume no laundry equipment or usage
        if not laundry.has_washer:
            return [0.0] * len(weekly_laundry_cycles_per_day)
        else:
            per_cycle = self.dhw_assumptions.efficient_washer_per_cycle if laundry.washer_efficient else self.dhw_assumptions.inefficient_washer_per_cycle
            daily_dhw = []
            for day_cycles in weekly_laundry_cycles_per_day:
                daily_dhw.append(day_cycles * per_cycle)
            return daily_dhw
        
    def weekly_dishwasher_dhw(self, weekly_dishwasher_cycles_per_day:list[int]) -> float:
        """Calculates weekly DHW usage based on dishwasher cycles."""
        #dishwasher = self.equipment.dishwasher
        dishwasher = sum(weekly_dishwasher_cycles_per_day) # If there are no cycles, we can assume no dishwasher equipment or usage
        if not dishwasher.has_dishwasher:
            return [0.0] * len(weekly_dishwasher_cycles_per_day)
        else:
            per_cycle = self.dhw_assumptions.efficient_dishwasher_per_cycle if dishwasher.dishwasher_efficient else self.dhw_assumptions.inefficient_dishwasher_per_cycle
            daily_dhw = []
            for day_cycles in weekly_dishwasher_cycles_per_day:
                daily_dhw.append(day_cycles * per_cycle)
            return daily_dhw
