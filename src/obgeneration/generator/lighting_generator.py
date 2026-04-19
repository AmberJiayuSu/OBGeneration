import obgeneration.model.lighting as Lighting
from obgeneration.generator.occupancy_generator import OccupancyGenerator, HouseholdOccupancyFractions
from obgeneration.generator.ob_utils import ScheduleUtils
from obgeneration.generator.results import LightingResult
from collections.abc import Sequence
from pydantic import BaseModel, ConfigDict, Field
from obgeneration.stochastic.distribution import Distribution
from obgeneration.stochastic.distribution_config import DistributionConfig



class LightingAssumptions(BaseModel):
    """Master configuration for lighting power assumptions."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    watts_per_m2_led: Distribution = Field(..., description="Design level lighting power per square meter for LED (W)")
    watts_per_m2_non_led: Distribution = Field(..., description="Design level lighting power per square meter for non-LED (W)")

    @classmethod
    def default(cls) -> "LightingAssumptions":
        """Returns the standard/default assumptions for lighting power."""
        return cls(
            watts_per_m2_led=DistributionConfig(
                dist_type="normal",
                params={"mean": 4.0, "std": 0.9, "lower": 0.5, "int": False},
            ).build(),
            watts_per_m2_non_led=DistributionConfig(
                dist_type="normal",
                params={"mean": 9.0, "std": 1.5, "lower": 1.0, "int": False},
            ).build(),
        )


class LightingGenerator:

    def __init__(self, lighting: Lighting.Lighting):
        self.lighting = lighting

    @staticmethod
    def generate_with_defaults(
        lighting: Lighting.Lighting,
        occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]],
    ) -> LightingResult:
        """Generate an annual lighting schedule from occupancy states.

        Derives the sleep mask from occupancy_states and returns the flattened
        annual lighting fraction schedule together with dimming metadata.
        """
        _, sleep_mask = OccupancyGenerator.active_sleep_mask(occupancy_states, 0.3)
        generator = LightingGenerator(lighting)
        dimming = generator.get_dimming()
        schedule = generator.lighting_annual_schedule(occupancy_states, sleep_mask)
        flattened_schedule = ScheduleUtils.flatten_schedule(schedule)
        if lighting.if_led:
            peak_value = generator.lighting_assumptions.watts_per_m2_led.sample()
        else:
            peak_value = generator.lighting_assumptions.watts_per_m2_non_led.sample()
        return LightingResult(
            peak_value=peak_value,
            schedule=flattened_schedule,
            dimming_enabled=dimming,
        )



    def lighting_annual_schedule(self, occupancy_annual_schedule:  Sequence[Sequence[HouseholdOccupancyFractions]], sleep_mask_annual: Sequence[Sequence[bool]]) -> list[list[float]]:
        """ Translates lighting usage pattern into a full annual schedule based on occupancy and sleep times."""
        annual_schedule = []
        for week_index in range(len(sleep_mask_annual)):
            weekly_sleep_mask = sleep_mask_annual[week_index]
            weekly_schedule = occupancy_annual_schedule[week_index]
            weekly_lighting_schedule = self.lighting_weekly_schedule(weekly_schedule,weekly_sleep_mask)
            annual_schedule.append(weekly_lighting_schedule)
        return annual_schedule
       
    
    def lighting_weekly_schedule(self, weekly_schedule: Sequence[HouseholdOccupancyFractions], sleep_mask_weekly: Sequence[bool]) -> list[float]:
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
