
from stochastic.distribution import Distribution
from model.occupancy import Occupancy
from model.lighting import Lighting
from model.equipment import Equipment
from model.hvac import HVAC
from generator.occupancy_generator import OccupancyGenerator, OccupancyAssumptions
from generator.lighting_generator import LightingGenerator
from generator.hvac_generator import HVACGenerator, HVACAssumptions
from generator.equipment_generator import EquipmentGenerator, EquipmentAssumptions
from generator.dhw_generator import DHWGenerator, DHWAssumptions
from pathlib import Path
import json
from pydantic import BaseModel, Field, ConfigDict


class ScheduleUtils:
    @staticmethod
    def weekly_index_day(index: int, resolution_mins: int) -> int:
        """ Returns the day of the week for a given index in a weekly schedule. """
        intervals_per_day = int(1440 / resolution_mins)
        return (index // intervals_per_day) % 7
    
    @staticmethod
    def weekly_index_hour(index: int, resolution_mins: int) -> float:
        """ Returns the hour of the day for a given index in a weekly schedule. """
        intervals_per_day = int(1440 / resolution_mins)
        index_in_day = index % intervals_per_day
        return (index_in_day * resolution_mins) / 60.0  
    
    @staticmethod
    def get_day_indices(day: int, resolution_mins: int) -> list[int]:
        """ Returns the list of indices for a specific day in a weekly schedule. """
        intervals_per_day = int(1440 / resolution_mins)
        start_index = day * intervals_per_day
        return list(range(start_index, start_index + intervals_per_day))

    @staticmethod
    def flatten_schedule(schedule):
        """ Flattens a schedule of lists into a single list. """
        return [item for sublist in schedule for item in sublist]
    

class OccupantBehavior(BaseModel):
    """ Represents the behavior profile of a household occupant, including occupancy, lighting, equipment, and HVAC. """
    model_config = ConfigDict(validate_assignment=True)
    num_occupants: int = Field(..., description="Number of occupants in the household.")
    occupancy_schedule: list[float] = Field(..., description="Occupancy schedule as a list of floats as a fraction of occupancy.")
    lighting_schedule: list[float] = Field(..., description="Lighting schedule as a list of fractions representing lighting usage.")
    lighting_if_dimming: float = Field(..., description="Indicates the probability of dimming being used.")
    equipment_schedule: list[float] = Field(..., description="Equipment schedule as a list of floats representing equipment usage for the household (W).")
    dhw_max_flow_rate_m3_per_s: float = Field(..., description="Maximum flow rate of domestic hot water in cubic meters per second.")
    dhw_schedule: list[float] = Field(..., description="Domestic hot water usage schedule as a list of fractions (relative to the max flow rate) representing flow rate.")    

    @staticmethod
    def aggregate_occupant_behavior(occupant_behaviors: list["OccupantBehavior"]) -> "OccupantBehavior":
        """ Aggregates occupant behaviors into a single occupant behavior. """
        num_occupants = sum(behavior.num_occupants for behavior in occupant_behaviors)
        occupancy_schedule = [0.0] * len(occupant_behaviors[0].occupancy_schedule)
        lighting_schedule = [0.0] * len(occupant_behaviors[0].lighting_schedule)
        equipment_schedule = [0.0] * len(occupant_behaviors[0].equipment_schedule)
        dhw_max_flow_rate_m3_per_schedule = [0.0] * len(occupant_behaviors[0].dhw_schedule)
        lighting_dimming_probability = 0.0

        for behavior in occupant_behaviors:
            lighting_dimming_probability += behavior.lighting_if_dimming * behavior.num_occupants / num_occupants
            for i in range(len(behavior.occupancy_schedule)):
                occupancy_schedule[i] += behavior.occupancy_schedule[i] * behavior.num_occupants / num_occupants
                equipment_schedule[i] += behavior.equipment_schedule[i] * behavior.num_occupants
                dhw_max_flow_rate_m3_per_schedule[i] += behavior.dhw_max_flow_rate_m3_per_s * behavior.dhw_schedule[i]
                lighting_schedule[i] += behavior.lighting_schedule[i] * behavior.num_occupants  / num_occupants

        dhw_max_flow_rate_m3_per_s = max(dhw_max_flow_rate_m3_per_schedule)
        dhw_schedule = [dhw / dhw_max_flow_rate_m3_per_s for dhw in dhw_max_flow_rate_m3_per_schedule]
            
        return OccupantBehavior(
            num_occupants=sum(behavior.num_occupants for behavior in occupant_behaviors),
            occupancy_schedule=occupancy_schedule,
            lighting_schedule=lighting_schedule,
            lighting_if_dimming=lighting_dimming_probability,
            equipment_schedule=equipment_schedule,
            dhw_max_flow_rate_m3_per_s=dhw_max_flow_rate_m3_per_s,
            dhw_schedule=dhw_schedule,
        )



class Occupant(BaseModel):
    occupancy: Occupancy = Field(...)
    lighting: Lighting = Field(...)
    equipment: Equipment = Field(...)
    hvac: HVAC = Field(...)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Occupant":
        """Load occupancy behavior from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            occupancy=Occupancy.model_validate(data["occupancy"]),
            lighting=Lighting.model_validate(data["lighting"]),
            equipment=Equipment.model_validate(data["equipment"]),
            hvac=HVAC.model_validate(data["hvac"])
        )
    
    def to_json_file(self, path: str | Path) -> None:
        """Save occupancy behavior to a JSON file."""
        data = {
            "occupancy": self.occupancy.model_dump(),
            "lighting": self.lighting.model_dump(),
            "equipment": self.equipment.model_dump(),
            "hvac": self.hvac.model_dump()
        }
        Path(path).write_text(json.dumps(data, indent=4))


    def to_OB_annual(self, resolution_mins) -> OccupantBehavior:
        """ Generate annual occupancy behavior schedules. """
        occ_gen = OccupancyGenerator( self.occupancy, OccupancyAssumptions.default(), resolution_mins)
        occupancy_schedule,sleep_schedule = occ_gen.household_annual_schedule()
        occupancy_mask, active_mask, sleep_mask = occ_gen.get_annual_mask(occupancy_schedule, sleep_schedule)

        lighting_gen = LightingGenerator(self.lighting)
        if_dimming = lighting_gen.get_dimming()
        lighting_schedule = lighting_gen.lighting_annual_schedule(occupancy_mask, sleep_mask)

        equipment_gen = EquipmentGenerator(self.equipment,active_mask,sleep_mask, occupancy_schedule, self.occupancy.num_occupants, resolution_mins, EquipmentAssumptions.default())
        equipment_schedule, laundry_cycles, dishwasher_cycles = equipment_gen.equipment_annual_schedule()

        dhw_gen = DHWGenerator( DHWAssumptions.default(), self.equipment, self.occupancy.num_occupants, laundry_cycles, dishwasher_cycles, resolution_mins)
        flow_rate, dhw_schedule = dhw_gen.dhw_annual_schedule()
        return OccupantBehavior(
            num_occupants=self.occupancy.num_occupants,
            occupancy_schedule=ScheduleUtils.flatten_schedule(occupancy_schedule),
            lighting_schedule=ScheduleUtils.flatten_schedule(lighting_schedule),
            lighting_if_dimming= 1.0 if if_dimming else 0.0,
            equipment_schedule=ScheduleUtils.flatten_schedule(equipment_schedule),
            dhw_max_flow_rate_m3_per_s=flow_rate,
            dhw_schedule=ScheduleUtils.flatten_schedule(dhw_schedule)
        )
    
    @staticmethod
    def multiple_to_OB_annual(occupants: dict["Occupant", int], resolution_mins: int) -> OccupantBehavior:
        """ Generate annual aggregated occupancy behavior schedules for multiple occupants. """
        behaviors = []
        for occupant, count in occupants.items():
            behavior = occupant.to_OB_annual(resolution_mins)
            for _ in range(count):
                behaviors.append(behavior)

        


