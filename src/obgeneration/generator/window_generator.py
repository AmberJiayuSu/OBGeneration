from obgeneration.stochastic.distribution import Distribution, CategoricalDistribution
from pydantic import BaseModel, Field, ConfigDict
from obgeneration.stochastic.distribution_config import DistributionConfig
import json
from pathlib import Path
from obgeneration.model.window import Window, WindowOpeningBehavior
from typing import Tuple


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
            heating_season=(42, 19),  # Weeks 42-20 (Oct 15 - May 13)
            cooling_season=(25, 35),  # Weeks 25-10 (Jun 18 - Sep 2)
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
    """Generates the annual opening fraction schedule for window, 
       ZoneVentilation:WindandStackOpenArea - Opening Area Fraction Schedule Name"""
    def __init__(self, window: Window, assumptions: WindowAssumptions, resolution_mins: int = 15):
        self.window = window
        self.assumptions = assumptions
        self.resolution_mins = resolution_mins

    def window_annual_schedule(self, lowest_cooling_setpoint, highest_heating_setpoint, occupancy_active_mask: list[list[bool]]) -> list[list[float]]:
        raise NotImplementedError("Window schedule generation is not yet implemented. This will produce a schedule of window opening fractions based on the specified window behavior and assumptions, coordinated with occupancy patterns and HVAC setpoints.")
        opening_fraction_schedule = [[0.0 for _ in row] for row in occupancy_active_mask] #start with all off
        max_outdoor_temp_schedule = [[lowest_cooling_setpoint for _ in row] for row in occupancy_active_mask] 
        min_outdoor_temp_schedule = [[highest_heating_setpoint for _ in row] for row in occupancy_active_mask]
        if self.window.heating_season is not None:
            if self.window.heating_season == WindowOpeningBehavior.NATURAL_VENTILATION:
                for i in range(len(opening_fraction_schedule)):
                    for j in range(len(opening_fraction_schedule[i])):
                        if occupancy_active_mask[i][j]:
                            opening_fraction_schedule[i][j] = self.assumptions.natural_ventilation_opening_fraction
            elif self.window.heating_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
                for i in range(len(opening_fraction_schedule)):
                    if WindowGenerator.in_week_range(i, self.assumptions.heating_season[0], self.assumptions.heating_season[1]):
                        for j in range(len(opening_fraction_schedule[i])):
                            if occupancy_active_mask[i][j]:
                                opening_fraction_schedule[i][j] = self.assumptions.occasionally_open_fraction
                                min_outdoor_temp_schedule[i][j] = -100
            elif self.window.heating_season == WindowOpeningBehavior.LONG_OPEN:
                for i in range(len(opening_fraction_schedule)):
                    if WindowGenerator.in_week_range(i, self.assumptions.heating_season[0], self.assumptions.heating_season[1]):
                        for j in range(len(opening_fraction_schedule[i])):
                            if occupancy_active_mask[i][j]:
                                opening_fraction_schedule[i][j] = self.assumptions.long_open_fraction
                                min_outdoor_temp_schedule[i][j] = -100
        if self.window.cooling_season is not None:
            pass
        if self.window.shoulder_season is not None:
            for i in range(len(opening_fraction_schedule)):
                for j in range(len(opening_fraction_schedule[i])):
                    if occupancy_active_mask[i][j]:
                        opening_fraction_schedule[i][j] = self.assumptions.shoulder_season_opening_fraction
        # if self.window.heating_season is not None:
        #     if self.window.heating_season == WindowOpeningBehavior.FREQUENTLY_OPEN:
        #         for i in range(len(schedule)):
        #             if WindowGenerator.in_week_range(i, self.assumptions.heating_season[0], self.assumptions.heating_season[1]):
        #                 for j in range(len(schedule[i])):
        #                     if occupancy_active_mask[i][j]:
        #                         schedule[i][j] = self.assumptions.small_open_fraction
        #     elif self.window.heating_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
        #         slots_per_day = 24 * (60 // self.resolution_mins)
        #         for i in range(len(schedule)):
        #             if WindowGenerator.in_week_range(i, self.assumptions.heating_season[0], self.assumptions.heating_season[1]):
        #                 day_range = 1 if i == 52 else 7
        #                 for d in range(day_range):
        #                     num_of_times = round(self.assumptions.short_opening_times.sample())
        #                     day_start = d * slots_per_day
        #                     day_end = day_start + slots_per_day
        #                     occupancy_mask_float = [1.0 if occupancy_active_mask[i][j] else 0.0 for j in range(day_start, day_end)]
        #                     start_time_dist = CategoricalDistribution(occupancy_mask_float)
        #                     cnt = 0
        #                     max_attempts = num_of_times * 10 + 50
        #                     attempts = 0
        #                     while cnt < num_of_times and attempts < max_attempts:
        #                         attempts += 1
        #                         start_time = start_time_dist.sample()
        #                         start_time_index = day_start + start_time
        #                         duration = self.assumptions.short_open_duration.sample()
        #                         duration_index = max(1, round(duration / self.resolution_mins))
        #                         if start_time_index + duration_index > day_end:
        #                             continue
        #                         for j in range(start_time_index, start_time_index + duration_index):
        #                             schedule[i][j] = self.assumptions.large_open_fraction
        #                         cnt += 1
        #                         start_time_dist.update_probabilities_by_factor({0.0: list(range(start_time, start_time + duration_index))})

        # if self.window.cooling_season is not None:
        #     if self.window.cooling_season == WindowOpeningBehavior.FREQUENTLY_OPEN:
        #         for i in range(len(schedule)):
        #             if WindowGenerator.in_week_range(i, self.assumptions.cooling_season[0], self.assumptions.cooling_season[1]):
        #                 for j in range(len(schedule[i])):
        #                     if occupancy_active_mask[i][j]:
        #                         schedule[i][j] = self.assumptions.small_open_fraction
        #     elif self.window.cooling_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
        #         slots_per_day = 24 * (60 // self.resolution_mins)
        #         for i in range(len(schedule)):
        #             if WindowGenerator.in_week_range(i, self.assumptions.cooling_season[0], self.assumptions.cooling_season[1]):
        #                 day_range = 1 if i == 52 else 7
        #                 for d in range(day_range):
        #                     num_of_times = round(self.assumptions.short_opening_times.sample())
        #                     day_start = d * slots_per_day
        #                     day_end = day_start + slots_per_day
        #                     occupancy_mask_float = [1.0 if occupancy_active_mask[i][j] else 0.0 for j in range(day_start, day_end)]
        #                     start_time_dist = CategoricalDistribution(occupancy_mask_float)
        #                     cnt = 0
        #                     max_attempts = num_of_times * 10 + 50
        #                     attempts = 0
        #                     while cnt < num_of_times and attempts < max_attempts:
        #                         attempts += 1
        #                         start_time = start_time_dist.sample()
        #                         start_time_index = day_start + start_time
        #                         duration = self.assumptions.short_open_duration.sample()
        #                         duration_index = max(1, round(duration / self.resolution_mins))
        #                         if start_time_index + duration_index > day_end:
        #                             continue
        #                         for j in range(start_time_index, start_time_index + duration_index):
        #                             schedule[i][j] = self.assumptions.large_open_fraction
        #                         cnt += 1
        #                         start_time_dist.update_probabilities_by_factor({0.0: list(range(start_time, start_time + duration_index))})
        # if self.window.shoulder_season is not None:
        #     if self.window.shoulder_season == WindowOpeningBehavior.FREQUENTLY_OPEN:
        #         for i in range(len(schedule)):
        #             for season in self.assumptions.shoulder_season:
        #                 if WindowGenerator.in_week_range(i, season[0], season[1]):
        #                     for j in range(len(schedule[i])):
        #                         if occupancy_active_mask[i][j]:
        #                             schedule[i][j] = self.assumptions.small_open_fraction
        #     elif self.window.shoulder_season == WindowOpeningBehavior.OCCASIONALLY_OPEN:
        #         slots_per_day = 24 * (60 // self.resolution_mins)
        #         for i in range(len(schedule)):
        #             for season in self.assumptions.shoulder_season:
        #                 if WindowGenerator.in_week_range(i, season[0], season[1]):
        #                     day_range = 1 if i == 52 else 7
        #                     for d in range(day_range):
        #                         num_of_times = round(self.assumptions.short_opening_times.sample())
        #                         day_start = d * slots_per_day
        #                         day_end = day_start + slots_per_day
        #                         occupancy_mask_float = [1.0 if occupancy_active_mask[i][j] else 0.0 for j in range(day_start, day_end)]
        #                         start_time_dist = CategoricalDistribution(occupancy_mask_float)
        #                         cnt = 0
        #                         max_attempts = num_of_times * 10 + 50
        #                         attempts = 0
        #                         while cnt < num_of_times and attempts < max_attempts:
        #                             attempts += 1
        #                             start_time = start_time_dist.sample()
        #                             start_time_index = day_start + start_time
        #                             duration = self.assumptions.short_open_duration.sample()
        #                             duration_index = max(1, round(duration / self.resolution_mins))
        #                             if start_time_index + duration_index > day_end:
        #                                 continue
        #                             for j in range(start_time_index, start_time_index + duration_index):
        #                                 schedule[i][j] = self.assumptions.large_open_fraction
        #                             cnt += 1
        #                             start_time_dist.update_probabilities_by_factor({0.0: list(range(start_time, start_time + duration_index))})
        # return schedule

    

    @staticmethod
    def in_week_range(week: int, start_week: int, end_week: int) -> bool:
        """Check if week number is within a week range (inclusive)."""
        if start_week <= end_week:
            return start_week <= week <= end_week
        else:
            return week >= start_week or week <= end_week
    
    