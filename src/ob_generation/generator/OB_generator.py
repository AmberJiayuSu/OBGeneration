from ob_generation.model.occupant_profile import Occupant
from ob_generation.generator.occupancy_generator import OccupancyGenerator, ClusterAssumptions
from ob_generation.generator.equipment_generator import EquipmentGenerator, EquipmentAssumptions
from ob_generation.generator.lighting_generator import LightingGenerator
from ob_generation.generator.dhw_generator import DHWGenerator, DHWAssumptions
from ob_generation.generator.hvac_generator import HVACGenerator, HVACAssumptions
from ob_generation.generator.ob_utils import ScheduleUtils
from ob_generation.generator.window_generator import WindowGenerator, WindowAssumptions
from pydantic import BaseModel, Field, ConfigDict

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional




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
    heating_setpoint: Optional[list[float]] = Field(..., description="Heating setpoint temperature in degrees Celsius.")
    cooling_setpoint: Optional[list[float]] = Field(..., description="Cooling setpoint temperature in degrees Celsius.")  
    window_schedule: Optional[list[float]] = Field(..., description="Window schedule as a list of lists of floats representing window opening fraction.")

    # @staticmethod
    # def aggregate_occupant_behavior(occupant_behaviors: list["OccupantBehavior"]) -> "OccupantBehavior":
    #     """ Aggregates occupant behaviors into a single occupant behavior. """
    #     num_occupants = sum(behavior.num_occupants for behavior in occupant_behaviors)
    #     occupancy_schedule = [0.0] * len(occupant_behaviors[0].occupancy_schedule)
    #     lighting_schedule = [0.0] * len(occupant_behaviors[0].lighting_schedule)
    #     equipment_schedule = [0.0] * len(occupant_behaviors[0].equipment_schedule)
    #     dhw_max_flow_rate_m3_per_schedule = [0.0] * len(occupant_behaviors[0].dhw_schedule)
    #     lighting_dimming_probability = 0.0

    #     for behavior in occupant_behaviors:
    #         lighting_dimming_probability += behavior.lighting_if_dimming * behavior.num_occupants / num_occupants
    #         for i in range(len(behavior.occupancy_schedule)):
    #             occupancy_schedule[i] += behavior.occupancy_schedule[i] * behavior.num_occupants / num_occupants
    #             equipment_schedule[i] += behavior.equipment_schedule[i] * behavior.num_occupants
    #             dhw_max_flow_rate_m3_per_schedule[i] += behavior.dhw_max_flow_rate_m3_per_s * behavior.dhw_schedule[i]
    #             lighting_schedule[i] += behavior.lighting_schedule[i] * behavior.num_occupants  / num_occupants

    #     dhw_max_flow_rate_m3_per_s = max(dhw_max_flow_rate_m3_per_schedule)
    #     dhw_schedule = [dhw / dhw_max_flow_rate_m3_per_s for dhw in dhw_max_flow_rate_m3_per_schedule]
            
    #     return OccupantBehavior(
    #         num_occupants=sum(behavior.num_occupants for behavior in occupant_behaviors),
    #         occupancy_schedule=occupancy_schedule,
    #         lighting_schedule=lighting_schedule,
    #         lighting_if_dimming=lighting_dimming_probability,
    #         equipment_schedule=equipment_schedule,
    #         dhw_max_flow_rate_m3_per_s=dhw_max_flow_rate_m3_per_s,
    #         dhw_schedule=dhw_schedule,
    #     )

    @staticmethod
    def to_OB_annual(resolution_mins: int, occupant_profile: Occupant) -> "OccupantBehavior":
        """ Generate annual occupancy behavior schedules. """
        occ_gen = OccupancyGenerator( occupant_profile.occupancy, ClusterAssumptions.default(), resolution_mins)
        occupancy_states = occ_gen.generate()
        occ_schedule = OccupancyGenerator.to_occupancy_schedule(occupancy_states)
        active_mask,sleep_mask = OccupancyGenerator.active_sleep_mask(occ_schedule)
        lighting_gen = LightingGenerator(occupant_profile.lighting)
        if_dimming = lighting_gen.get_dimming()
        lighting_schedule = lighting_gen.lighting_annual_schedule( sleep_mask)

        equipment_gen = EquipmentGenerator(occupant_profile.equipment,occupancy_states, occupant_profile.occupancy.num_occupants, resolution_mins, EquipmentAssumptions.default())
        equipment_schedule, laundry_cycles, dishwasher_cycles = equipment_gen.equipment_annual_schedule()

        dhw_gen = DHWGenerator( DHWAssumptions.default(), occupant_profile.equipment, occupant_profile.occupancy.num_occupants, laundry_cycles, dishwasher_cycles, resolution_mins)
        flow_rate, dhw_schedule = dhw_gen.dhw_annual_schedule()

        hvac_gen = HVACGenerator( occupant_profile.hvac, HVACAssumptions.default())
        heating_setpoint = hvac_gen.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling_setpoint = hvac_gen.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        window_gen = WindowGenerator(occupant_profile.window, WindowAssumptions.default(), resolution_mins)
        window_schedule = window_gen.window_annual_schedule(active_mask)

        return OccupantBehavior(
            num_occupants=occupant_profile.occupancy.num_occupants,
            occupancy_schedule=occ_schedule,
            lighting_schedule=ScheduleUtils.flatten_schedule(lighting_schedule),
            lighting_if_dimming= 1.0 if if_dimming else 0.0,
            equipment_schedule=ScheduleUtils.flatten_schedule(equipment_schedule),
            dhw_max_flow_rate_m3_per_s=flow_rate,
            dhw_schedule=ScheduleUtils.flatten_schedule(dhw_schedule),
            heating_setpoint=ScheduleUtils.flatten_schedule(heating_setpoint) if heating_setpoint is not None else None,
            cooling_setpoint=ScheduleUtils.flatten_schedule(cooling_setpoint) if cooling_setpoint is not None else None,
            window_schedule=ScheduleUtils.flatten_schedule(window_schedule) if window_schedule is not None else None
        )
    

    # @staticmethod
    # def multiple_to_OB_annual(occupants: dict["Occupant", int], resolution_mins: int) -> "OccupantBehavior":
    #     """ Generate annual aggregated occupancy behavior schedules for multiple occupants. """
    #     behaviors = []
        
    #     total_occupants = 0
    #     total_cnt = 0
    #     total_dimming = 0
    #     length = 365 * 24 * (60 // resolution_mins)
    #     all_occupancy_schedules = [0.0] * length
    #     all_equipment_schedules = [0.0] * length
    #     all_dhw_schedules = [0.0] * length
    #     all_lighting_schedules = [0.0] * length

    #     for occupant, count in occupants.items():
    #         total_cnt += count
    #         for _ in range(count):
    #             total_occupants += occupant.occupancy.num_occupants
    #             occ_gen = OccupancyGenerator( occupant.occupancy, OccupancyAssumptions.default(), resolution_mins)
    #             occupancy_schedule,sleep_schedule = occ_gen.household_annual_schedule()
    #             occupancy_mask, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy_schedule, sleep_schedule)
    #             equipment_gen = EquipmentGenerator(occupant.equipment,active_mask,sleep_mask, occupancy_schedule, occupant.occupancy.num_occupants, resolution_mins, EquipmentAssumptions.default())
    #             equipment_schedule, laundry_cycles, dishwasher_cycles = equipment_gen.equipment_annual_schedule()
    #             dhw_gen = DHWGenerator( DHWAssumptions.default(), occupant.equipment, occupant.occupancy.num_occupants, laundry_cycles, dishwasher_cycles, resolution_mins)
    #             flow_rate, dhw_schedule = dhw_gen.dhw_annual_schedule()
    #             lighting_gen = LightingGenerator(occupant.lighting)
    #             if_dimming = lighting_gen.get_dimming()
    #             lighting_schedule = lighting_gen.lighting_annual_schedule(occupancy_mask, sleep_mask)

    #             total_dimming += (1.0 if if_dimming else 0.0) * occupant.occupancy.num_occupants
    #             for i in range(len(occupancy_schedule)):
    #                 all_occupancy_schedules[i] += ScheduleUtils.flatten_schedule(occupancy_schedule)[i] * occupant.occupancy.num_occupants
    #                 all_equipment_schedules[i] += ScheduleUtils.flatten_schedule(equipment_schedule)[i] 
    #                 all_dhw_schedules[i] += ScheduleUtils.flatten_schedule(dhw_schedule)[i] * flow_rate
    #                 all_lighting_schedules[i] += ScheduleUtils.flatten_schedule(lighting_schedule)[i] 


    #     dimming_probability = total_dimming / total_cnt

    #     normalized_lighting_schedule = [light / total_cnt for light in all_lighting_schedules]

    #     normalized_occupancy_schedule = [occ / total_occupants for occ in all_occupancy_schedules]

    #     max_dhw_flow_rate = max(all_dhw_schedules)
    #     if max_dhw_flow_rate == 0:
    #         normalized_dhw_schedule = all_dhw_schedules
    #     else:
    #         normalized_dhw_schedule = [flow / max_dhw_flow_rate for flow in all_dhw_schedules]
            
        
    #     return OccupantBehavior.aggregate_occupant_behavior(behaviors)
        




