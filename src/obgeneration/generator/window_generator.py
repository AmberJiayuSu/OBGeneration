from collections.abc import Sequence
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from obgeneration.generator.results import HVACResult, WindowResult
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.model.window import Window
from obgeneration.stochastic.distribution import Distribution
from obgeneration.stochastic.distribution_config import DistributionConfig


WeekRange = Tuple[int, int]  # (start_week, end_week) where weeks are 1-52


class WindowAssumptions(BaseModel):
    """Window operation assumptions."""
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    heating_season: WeekRange = Field(..., description="Start and end week numbers for heating season (inclusive, weeks 1-52).")
    cooling_season: WeekRange = Field(..., description="Start and end week numbers for cooling season (inclusive, weeks 1-52).")
    shoulder_season: list[WeekRange] = Field(..., description="Start and end week numbers for shoulder seasons (inclusive, weeks 1-52).")
    short_open_duration: Distribution = Field(..., description="Short open duration in minutes.")
    short_opening_times: Distribution = Field(..., description="The number of times the window is opened for short duration in a day.")
    long_open_duration: Distribution = Field(..., description="Long open duration in minutes.")
    long_opening_times: Distribution = Field(..., description="The number of times the window is opened for long duration in a week.")
    natural_ventilation_opening_fraction: float = Field(..., ge=0.0, le=1.0, description="Opening fraction for natural ventilation behavior.")
    occasionally_open_fraction: float = Field(..., ge=0.0, le=1.0, description="Opening fraction for occasionally open behavior.")
    long_open_fraction: float = Field(..., ge=0.0, le=1.0, description="Opening fraction for long open behavior.")


    @classmethod
    def from_json_file(cls, path: str | Path) -> "WindowAssumptions":
        """Load window assumptions from a JSON file
           Weeks are numbered 1-52, where week 1 starts on January 1st.
        """
        data = json.loads(Path(path).read_text())
        
        def parse_week_range(range_list: list[int]) -> WeekRange:
            """Parse a season range from [start_week, end_week] list."""
            start_week, end_week = range_list[0], range_list[1]
            if not (1 <= start_week <= 52 and 1 <= end_week <= 52):
                raise ValueError(f"Week numbers must be between 1 and 52, got {start_week} and {end_week}")
            return (start_week, end_week)
        
        data['heating_season'] = parse_week_range(data['heating_season'])
        data['cooling_season'] = parse_week_range(data['cooling_season'])
        data['shoulder_season'] = [parse_week_range(season) for season in data['shoulder_season']]
        data['short_open_duration'] = DistributionConfig(data['short_open_duration']).build()
        data['short_opening_times'] = DistributionConfig(data['short_opening_times']).build()
        data['long_open_duration'] = DistributionConfig(data['long_open_duration']).build()
        data['long_opening_times'] = DistributionConfig(data['long_opening_times']).build()
        data['natural_ventilation_opening_fraction'] = data['natural_ventilation_opening_fraction']
        data['occasionally_open_fraction'] = data['occasionally_open_fraction']
        data['long_open_fraction'] = data['long_open_fraction']
        return cls(**data)


    @classmethod
    def default(cls) -> "WindowAssumptions":
        """Returns the standard/default assumptions for window operation."""
        return cls(
            heating_season=(42, 19),  # Weeks 42-19
            cooling_season=(25, 35),  # Weeks 25-35
            shoulder_season=[(20, 24), (36, 41)],  # Weeks 20-24 and 36-41
            short_open_duration=DistributionConfig(dist_type="normal", params={"mean": 10, "std": 5, "lower": 0, "upper": 30, "int": False}).build(),
            short_opening_times=DistributionConfig(dist_type="normal", params={"mean": 2, "std": 1, "lower": 0, "upper": 3, "int": False}).build(),
            long_open_duration=DistributionConfig(dist_type="normal", params={"mean": 180, "std": 60, "lower": 0, "upper": 300, "int": False}).build(),
            long_opening_times=DistributionConfig(dist_type="normal", params={"mean": 1, "std": 0.5, "lower": 0, "upper": 3, "int": False}).build(),
            natural_ventilation_opening_fraction=0.5,
            occasionally_open_fraction=0.5,
            long_open_fraction=0.3
        )

class WindowGenerator:
    """Hook for an annual window opening-fraction schedule.

    `generate_result` is on the household call so a later implementation can
    fill `WindowResult` without changing the orchestrator. It returns `None`
    until that schedule is specified. The method does not draw from `rng`.
    """
    def __init__(self, window: Window, assumptions: WindowAssumptions, resolution_mins: int = 15):
        self.window = window
        self.assumptions = assumptions
        self.resolution_mins = resolution_mins

    @staticmethod
    def generate_result(
        window: Window,
        occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]],
        assumptions: WindowAssumptions,
        resolution_mins: int,
        rng: np.random.Generator | int,
        *,
        hvac: HVACResult | None = None,
        active_threshold: float = 0.51,
        away_threshold: float = 0.01,
    ) -> WindowResult | None:
        """Reserved entry for the opening-fraction schedule.

        Arguments are the inputs a later implementation should use: the
        seasonal behavior on `window`, `assumptions` (week ranges and opening
        fractions), household `occupancy_states`, the HVAC setpoints on
        `hvac`, and `resolution_mins`. `active_threshold` and `away_threshold`
        match the masks used for HVAC. `rng` is accepted so sampling can be
        added here without moving this call ahead of lighting or equipment.

        Returns `None`. The arguments are unused until the schedule exists.
        """
        del (
            window,
            occupancy_states,
            assumptions,
            resolution_mins,
            rng,
            hvac,
            active_threshold,
            away_threshold,
        )
        return None

    @staticmethod
    def generate_with_defaults(
        window: Window,
        occupancy_states: Sequence[Sequence[HouseholdOccupancyFractions]],
        resolution_mins: int,
        rng: np.random.Generator | int,
        *,
        hvac: HVACResult | None = None,
        active_threshold: float = 0.51,
        away_threshold: float = 0.01,
    ) -> WindowResult | None:
        """Same hook as `generate_result`, with packaged window assumptions."""
        return WindowGenerator.generate_result(
            window=window,
            occupancy_states=occupancy_states,
            assumptions=WindowAssumptions.default(),
            resolution_mins=resolution_mins,
            rng=rng,
            hvac=hvac,
            active_threshold=active_threshold,
            away_threshold=away_threshold,
        )

    def window_annual_schedule(
        self,
        occupancy_active_mask: list[list[bool]],
        lowest_cooling_setpoint: float | None = None,
        highest_heating_setpoint: float | None = None,
    ) -> list[list[float]] | None:
        """Instance hook kept for `OccupantBehavior.to_OB_annual`.

        Returns `None`. The first argument is the occupancy mask that call
        already passes; the setpoint arguments are optional so that call does
        not have to supply them.
        """
        del occupancy_active_mask, lowest_cooling_setpoint, highest_heating_setpoint
        return None

    @staticmethod
    def in_week_range(week: int, start_week: int, end_week: int) -> bool:
        """Check if week number is within a week range (inclusive).

        A range that wraps the year (`start_week` > `end_week`) includes
        both ends, so weeks 42 through 19 covers late year and early year.
        """
        if start_week <= end_week:
            return start_week <= week <= end_week
        else:
            return week >= start_week or week <= end_week
