
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import model.lighting as Lighting
import model.equipment as Equipment
import model.hvac as HVAC
import stochastic.translation_rule as TranslationRule
from pathlib import Path
import json

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
    
  

class ModelScheduleGenerator:

    def __init__(self, json_path: Path):
        self.json_path = json_path
        with open(json_path, 'r') as file:
            data = json.load(file)
        self.raw_data = data
        occupancy_data = data.get('occupancy', {})
        self.occupancy = Occupancy.Occupancy.model_validate(occupancy_data)
        lighting_data = data.get('lighting', {})
        self.lighting = Lighting.Lighting.model_validate(lighting_data)
        equipment_data = data.get('equipment', {})
        self.equipment = Equipment.Equipment.model_validate(equipment_data)
        hvac_data = data.get('hvac', {})
        self.hvac = HVAC.HVAC.model_validate(hvac_data)
        

    def generate_annual_schedule(self) -> dict:
        """ Generates a weekly schedule for occupancy, lighting, equipment, and HVAC. """
        annual_schedule = {
            "occupancy": [],
            "lighting": [],
            "equipment": [],
            "heating_setpoint": [],
            "cooling_setpoint": []
        }
        for _ in range(0, 53):
            occupancy_translator = OccupancyTranslator(self.occupancy)
            occupancy_schedule = occupancy_translator.translate_occupancy_presumption()
            annual_schedule["occupancy"].extend(occupancy_schedule)
            active_time_mask = occupancy_translator.get_occupied_active_mask()
            sleep_mask = occupancy_translator.get_occupied_sleep_mask()
            occupied_time_mask = occupancy_translator.get_occupied_mask()
            lighting_translator = LightingTranslator(self.lighting, sleep_mask, occupancy_schedule)
            lighting_schedule = lighting_translator.translate_lighting_schedule()
            annual_schedule["lighting"].extend(lighting_schedule)
            equipment_translator = EquipmentTranslator(self.equipment, active_time_mask)
            equipment_schedule = equipment_translator.get_equipment_usage_schedule()
            annual_schedule["equipment"].extend(equipment_schedule)
            hvac_translator = HVACTranslator(self.hvac, sleep_mask, occupied_time_mask)
            heating_schedule = hvac_translator.translate_heating_setpoint_schedule()
            annual_schedule["heating_setpoint"].extend(heating_schedule)
            cooling_schedule = hvac_translator.translate_cooling_setpoint_schedule()
            annual_schedule["cooling_setpoint"].extend(cooling_schedule)
        for key in annual_schedule:
            if len(annual_schedule[key]) > 8760:
                annual_schedule[key] = annual_schedule[key][:8760]
        return annual_schedule
    
    def get_dimming(self) -> bool:
        """ Determines if lighting dimming is used. """
        lighting_translator = LightingTranslator(self.lighting, [], [])
        return lighting_translator.get_dimming(self.lighting)