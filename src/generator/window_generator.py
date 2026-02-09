from stochastic.distribution import Distribution, UniformDistribution, CategoricalDistribution
from generator.ob_utils import ScheduleUtils
import model.equipment as Equipment
from pydantic import BaseModel, Field, ConfigDict
from stochastic.distribution_config import DistributionConfig
import json
from pathlib import Path
from datetime import  date, timedelta
from model.window import Window, WindowOpeningBehavior
from typing import Tuple, List


WeekRange = Tuple[int, int]  # (start_week, end_week) where weeks are 1-52


class WindowAssumptions(BaseModel):
    """Window operation assumptions."""
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    heating_season: WeekRange = Field(..., description="Start and end week numbers for heating season (inclusive, weeks 1-52).")
    cooling_season: WeekRange = Field(..., description="Start and end week numbers for cooling season (inclusive, weeks 1-52).")
    shoulder_season: list[WeekRange] = Field(..., description="Start and end week numbers for shoulder seasons (inclusive, weeks 1-52).")
    short_open_duration: Distribution = Field(..., description="Short open duration in minutes.")
    short_opening_times: Distribution = Field(..., description="The number of times the window is opened for short duration in a day.")
    small_open_fraction: float = Field(..., description="The fraction of the window that is opened for long duration.")
    large_open_fraction: float = Field(..., description="The fraction of the window that is opened for short duration.")


    @classmethod
    def from_json_file(cls, path: str | Path) -> "WindowAssumptions":
        """Load window assumptions from a JSON file.
        
        Expected JSON format:
        {
          "heating_season": [start_week, end_week],  // e.g., [42, 19] for weeks 42-19
          "cooling_season": [start_week, end_week],  // e.g., [25, 35] for weeks 25-35
          "shoulder_season": [                        // list of shoulder season ranges
            [start_week, end_week],                  // e.g., [21, 24] for weeks 21-24
            [start_week, end_week]                   // e.g., [36, 40] for weeks 36-40
          ],
          "short_open_duration": {...},               // DistributionConfig dict
          "short_opening_times": {...},               // DistributionConfig dict
          "small_open_fraction": 0.05,
          "large_open_fraction": 0.5
        }
        
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
        return cls(**data)


    @classmethod
    def default(cls) -> "WindowAssumptions":
        """Returns the standard/default assumptions for window operation."""
        return cls(
            heating_season=(42, 19),  # Weeks 42-20 (Oct 15 - May 13)
            cooling_season=(25, 35),  # Weeks 25-10 (Jun 18 - Sep 2)
            shoulder_season=[(20, 24), (36, 41)],  # Weeks 20-24 and 36-41
            short_open_duration=DistributionConfig(dist_type="normal", params={"mean": 10, "std": 5, "lower": 0, "upper": 30, "int": False}).build(),
            short_opening_times=DistributionConfig(dist_type="normal", params={"mean": 2, "std": 1, "lower": 0, "upper": 3, "int": False}).build(),
            small_open_fraction=0.05,
            large_open_fraction=0.5
        )

class WindowGenerator:
    """Generates the annual opening fraction schedule for window, 
       ZoneVentilation:WindandStackOpenArea - Opening Area Fraction Schedule Name"""
    def __init__(self, window: Window, assumptions: WindowAssumptions, resolution_mins: int = 15):
        self.window = window
        self.assumptions = assumptions
        self.resolution_mins = resolution_mins

    

    def window_annual_schedule(self, occupancy_active_mask: list[list[bool]]) -> list[list[float]]:
        schedule = [[0.0 for _ in row] for row in occupancy_active_mask]
        if self.window.heating_season is not None:
            if self.window.heating_season == WindowOpeningBehavior.FREQUENTLY_OPEN:
                for i in range(len(schedule)):
                    if WindowGenerator.in_week_range(i, self.assumptions.heating_season[0], self.assumptions.heating_season[1]):
                        for j in range(len(schedule[i])):
                            if occupancy_active_mask[i][j]:
                                schedule[i][j] = self.assumptions.small_open_fraction
            elif self.window.heating_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
                for i in range(len(schedule)):
                    if WindowGenerator.in_week_range(i, self.assumptions.heating_season[0], self.assumptions.heating_season[1]):
                        if i == 53:
                            day_range = 1
                        else:
                            day_range = 7
                        for d in range(day_range):
                            num_of_times = self.assumptions.short_opening_times.sample()
                            occupancy_mask_float = [1.0 if occupancy_active_mask[i][j] else 0.0 for j in range(len(schedule[i]))]
                            start_time_dist = CategoricalDistribution(occupancy_mask_float)
                            cnt = 0
                            while cnt < num_of_times:
                                start_time = start_time_dist.sample()
                                start_time_index = d * 24 * (60 // self.resolution_mins) + start_time
                                duration = self.assumptions.short_open_duration.sample()
                                duration_index = min(1, round(duration / self.resolution_mins))
                                if start_time_index + duration_index > len(schedule[i]):
                                    continue
                                for j in range(start_time_index, start_time_index + duration_index):
                                    schedule[i][j] = self.assumptions.large_open_fraction
                                cnt += 1
                                start_time_dist.update_probabilities_by_factor({0.0, range(start_time, start_time + duration_index)})

        if self.window.cooling_season is not None:
            if self.window.cooling_season == WindowOpeningBehavior.FREQUENTLY_OPEN:
                for i in range(len(schedule)):
                    if WindowGenerator.in_week_range(i, self.assumptions.cooling_season[0], self.assumptions.cooling_season[1]):
                        for j in range(len(schedule[i])):
                            if occupancy_active_mask[i][j]:
                                schedule[i][j] = self.assumptions.small_open_fraction
            elif self.window.cooling_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
                for i in range(len(schedule)):
                    if WindowGenerator.in_week_range(i, self.assumptions.cooling_season[0], self.assumptions.cooling_season[1]):
                        if i == 53:
                            day_range = 1
                        else:
                            day_range = 7
                        for d in range(day_range):
                            num_of_times = self.assumptions.short_opening_times.sample()
                            occupancy_mask_float = [1.0 if occupancy_active_mask[i][j] else 0.0 for j in range(len(schedule[i]))]
                            start_time_dist = CategoricalDistribution(occupancy_mask_float)
                            cnt = 0
                            while cnt < num_of_times:
                                start_time = start_time_dist.sample()
                                start_time_index = d * 24 * (60 // self.resolution_mins) + start_time
                                duration = self.assumptions.short_open_duration.sample()
                                duration_index = min(1, round(duration / self.resolution_mins))
                                if start_time_index + duration_index > len(schedule[i]):
                                    continue
                                for j in range(start_time_index, start_time_index + duration_index):
                                    schedule[i][j] = self.assumptions.large_open_fraction
                                cnt += 1
                                start_time_dist.update_probabilities_by_factor({0.0, range(start_time, start_time + duration_index)})
        if self.window.shoulder_season is not None:
            if self.window.shoulder_season == WindowOpeningBehavior.FREQUENTLY_OPEN:
                for i in range(len(schedule)):
                    for season in self.assumptions.shoulder_season:
                        if WindowGenerator.in_week_range(i, season[0], season[1]):
                            for j in range(len(schedule[i])):
                                if occupancy_active_mask[i][j]:
                                    schedule[i][j] = self.assumptions.small_open_fraction
            elif self.window.shoulder_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
                for i in range(len(schedule)):
                    for season in self.assumptions.shoulder_season:
                        if WindowGenerator.in_week_range(i, season[0], season[1]):
                            if i == 53:
                                day_range = 1
                            else:
                                day_range = 7
                            for d in range(day_range):
                                num_of_times = self.assumptions.short_opening_times.sample()
                                occupancy_mask_float = [1.0 if occupancy_active_mask[i][j] else 0.0 for j in range(len(schedule[i]))]
                                start_time_dist = CategoricalDistribution(occupancy_mask_float)
                                cnt = 0
                                while cnt < num_of_times:
                                    start_time = start_time_dist.sample()
                                    start_time_index = d * 24 * (60 // self.resolution_mins) + start_time
                                    duration = self.assumptions.short_open_duration.sample()
                                    duration_index = min(1, round(duration / self.resolution_mins))
                                    if start_time_index + duration_index > len(schedule[i]):
                                        continue
                                    for j in range(start_time_index, start_time_index + duration_index):
                                        schedule[i][j] = self.assumptions.large_open_fraction
                                    cnt += 1
                                    start_time_dist.update_probabilities_by_factor({0.0, range(start_time, start_time + duration_index)})
        return schedule

    






    @staticmethod
    def in_week_range(week: int, start_week: int, end_week: int) -> bool:
        """Check if week number is within a week range (inclusive)."""
        if start_week <= end_week:
            return start_week <= week <= end_week
        else:
            return week >= start_week or week <= end_week
    
    