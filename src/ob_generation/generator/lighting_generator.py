import ob_generation.model.lighting as Lighting
from ob_generation.generator.occupancy_generator import OccupancyGenerator, HouseholdOccupancyFractions



# class LightingAssumptions(BaseModel):
#     """Master configuration for lighting power assumptions."""
#     watts_per_person_led: Distribution = Field(..., description="Design level lighting power per person for LED (W)")
#     watts_per_person_non_led: Distribution = Field(..., description="Design level lighting power per person for non-LED (W)")
    
#     @classmethod
#     def default(cls) -> "LightingAssumptions":
#         """Returns the standard/default assumptions for lighting power."""
#         return cls(
#             watts_per_person_led= DistributionConfig(dist_type="normal", params={"mean": 500.0, "std": 50.0, "lower": 0.0, "int": False}).build(),
#             watts_per_person_non_led= DistributionConfig(dist_type="normal", params={"mean": 800.0, "std": 100.0, "lower": 0.0, "int": False}).build()
#         )


class LightingGenerator:

    def __init__(self, lighting: Lighting.Lighting):
        self.lighting = lighting



    def lighting_annual_schedule(self, occupancy_annual_schedule:  list[list[HouseholdOccupancyFractions]], sleep_mask_annual: list[list[bool]]) -> list[list[float]]:
        """ Translates lighting usage pattern into a full annual schedule based on occupancy and sleep times."""
        annual_schedule = []
        for week_index in range(len(sleep_mask_annual)):
            weekly_sleep_mask = sleep_mask_annual[week_index]
            weekly_schedule = occupancy_annual_schedule[week_index]
            weekly_lighting_schedule = self.lighting_weekly_schedule(weekly_schedule,weekly_sleep_mask)
            annual_schedule.append(weekly_lighting_schedule)
        return annual_schedule
       
    
    def lighting_weekly_schedule(self, weekly_schedule: list[HouseholdOccupancyFractions], sleep_mask_weekly: list[bool]) -> list[float]:
        """ Translates lighting usage pattern into a full week schedule based on occupancy and sleep times."""
        schedule = []
        length = len(sleep_mask_weekly)

        if self.lighting.when_away:
            schedule = [occupancy.home for occupancy in weekly_schedule]
        else:
            schedule = [1.0] * length

        # Adjust for sleep times
        # Assumption: during sleep time, lighting usage is zero
        schedule = OccupancyGenerator.apply_to_mask(sleep_mask_weekly, schedule, 0.0)
        return schedule
        

    def get_dimming(self) -> bool:
        """ Determines if dimming is used based on usage pattern. """
        if self.lighting.when_daylight_bright:
            return True
        return False

    # def get_design_watts_per_person(self) -> float:
    #     """Returns the design level lighting power per person based on LED status."""
    #     if self.lighting.if_led:
    #         return self.lighting_assumptions.watts_per_person_led
    #     else:
    #         return self.lighting_assumptions.watts_per_person_non_led