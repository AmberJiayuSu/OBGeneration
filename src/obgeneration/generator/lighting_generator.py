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
                params={"mean": 2.5, "std": 0.8, "lower": 0.5, "int": False},
            ).build(),
            watts_per_m2_non_led=DistributionConfig(
                dist_type="normal",
                params={"mean": 9.0, "std": 2.0, "lower": 1.0, "int": False},
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
        schedule, max_day, min_day = generator.lighting_annual_schedule(occupancy_states, sleep_mask)
        flattened_schedule = ScheduleUtils.flatten_schedule(schedule)
        if lighting.if_led:
            peak_value = generator.lighting_assumptions.watts_per_m2_led.sample(rng)
        else:
            peak_value = generator.lighting_assumptions.watts_per_m2_non_led.sample(rng)
        return LightingResult(
            peak_value=peak_value,
            annual_schedule=flattened_schedule,
            dimming_enabled=dimming,
            summer_design_day_schedule=max_day,
            winter_design_day_schedule=min_day,
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



    def lighting_annual_schedule(self, occupancy_annual_schedule:  Sequence[Sequence[HouseholdOccupancyFractions]], sleep_mask_annual: Sequence[Sequence[bool]]) -> tuple[list[list[float]], list[float], list[float]]:
        """ Translates lighting usage pattern into a full annual schedule based on occupancy and sleep times, while also identifying the days with maximum and minimum average usage. """
        annual_schedule = []
        max_avg = float("-inf")
        min_avg = float("inf")
        max_day = None
        min_day = None
        bins_per_day = len(occupancy_annual_schedule[0]) // 7
        for week_index in range(len(sleep_mask_annual)):
            weekly_sleep_mask = sleep_mask_annual[week_index]
            weekly_schedule = occupancy_annual_schedule[week_index]
            weekly_lighting_schedule = self.lighting_weekly_schedule(weekly_schedule,weekly_sleep_mask)
            for day_index in range(7):
                day_schedule = weekly_lighting_schedule[day_index * bins_per_day : (day_index + 1) * bins_per_day]
                if len(day_schedule) != bins_per_day:
                    continue
                avg_usage = np.mean(day_schedule)
                if avg_usage > max_avg:
                    max_avg = avg_usage
                    max_day = day_schedule.copy()
                if avg_usage < min_avg:
                    min_avg = avg_usage
                    min_day = day_schedule.copy()
            annual_schedule.append(weekly_lighting_schedule)

        return annual_schedule, max_day, min_day
       
    
    def lighting_weekly_schedule(self, weekly_schedule: Sequence[HouseholdOccupancyFractions], sleep_mask_weekly: Sequence[bool]) -> list[float]:
        """ Translates lighting usage pattern into a full week schedule based on occupancy and sleep times."""
        schedule = []
        schedule = [occupancy.home for occupancy in weekly_schedule]
        
        # length = len(sleep_mask_weekly)
        # if self.lighting.when_away:
        #     schedule = [occupancy.home for occupancy in weekly_schedule]
        # else:
        #     schedule = [1.0] * length

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
