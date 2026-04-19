import obgeneration.model.lighting as Lighting
from obgeneration.generator.occupancy_generator import OccupancyGenerator
from obgeneration.generator.ob_utils import ScheduleUtils
from obgeneration.generator.results import LightingResult
from obgeneration.generator.types import HouseholdOccupancyFractions
from collections.abc import Sequence
import numpy as np
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

    def __init__(self, lighting: Lighting.Lighting, lighting_assumptions: LightingAssumptions = LightingAssumptions.default()):
        self.lighting = lighting
        self.lighting_assumptions = lighting_assumptions

    @staticmethod
    def generate_result(
        lighting: Lighting.Lighting,
        occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]],
        lighting_assumptions: LightingAssumptions,
        rng: np.random.Generator | int,
    ) -> LightingResult:
        """Generate an annual lighting result with explicit assumptions."""
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        _, sleep_mask = OccupancyGenerator.active_sleep_mask(occupancy_states, 0.3)
        generator = LightingGenerator(lighting, lighting_assumptions)
        dimming = generator.get_dimming()
        schedule = generator.lighting_annual_schedule(occupancy_states, sleep_mask)
        flattened_schedule = ScheduleUtils.flatten_schedule(schedule)
        if lighting.if_led:
            peak_value = generator.lighting_assumptions.watts_per_m2_led.sample(rng)
        else:
            peak_value = generator.lighting_assumptions.watts_per_m2_non_led.sample(rng)
        return LightingResult(
            peak_value=peak_value,
            schedule=flattened_schedule,
            dimming_enabled=dimming,
        )

    @staticmethod
    def generate_with_defaults(
        lighting: Lighting.Lighting,
        occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]],
        rng: np.random.Generator | int,
    ) -> LightingResult:
        """Generate an annual lighting result using default assumptions."""
        return LightingGenerator.generate_result(
            lighting=lighting,
            occupancy_states=occupancy_states,
            lighting_assumptions=LightingAssumptions.default(),
            rng=rng,
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
