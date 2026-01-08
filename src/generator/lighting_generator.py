import model.lighting as Lighting
from generator.occupancy_generator import OccupancyGenerator


class LightingGenerator:

    def __init__(self, lighting: Lighting.Lighting):
        self.lighting = lighting


    def lighting_annual_schedule(self, occupancy_mask_annual: list[list[bool]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates lighting usage pattern into a full annual schedule based on occupancy and sleep times."""
        annual_schedule = []
        for week_index in range(len(occupancy_mask_annual)):
            weekly_occupancy_mask = occupancy_mask_annual[week_index]
            weekly_sleep_mask = sleep_mask_annual[week_index]
            weekly_lighting_schedule = self.lighting_weekly_schedule(weekly_occupancy_mask, weekly_sleep_mask)
            annual_schedule.append(weekly_lighting_schedule)
        return annual_schedule
       
    
    def lighting_weekly_schedule(self, occupancy_mask_weekly: list[bool], sleep_mask_weekly: list[bool]) -> list[float]:
        """ Translates lighting usage pattern into a full week schedule based on occupancy and sleep times."""
        lighting = self.lighting
        schedule = []
        length = len(occupancy_mask_weekly)
        #full on always
        if not lighting.when_house_empty:
            schedule = [1.0] * length
        else:
            schedule = [1.0 if occ else 0.0 for occ in occupancy_mask_weekly]
        # Adjust for sleep times 
        # Assumption: during sleep time, lighting usage is zero
        schedule = OccupancyGenerator.revise_by_sleep(sleep_mask_weekly, schedule, 0.0)       
        return schedule
        

    def get_dimming(self) -> bool:
        """ Determines if dimming is used based on usage pattern. """
        if self.lighting.when_daylight_bright:
            return True
        return False