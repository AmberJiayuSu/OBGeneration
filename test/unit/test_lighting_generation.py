import pytest
import numpy as np

from obgeneration.model.lighting import Lighting
from obgeneration.model.occupancy import Occupancy
from obgeneration.generator.lighting_generator import LightingAssumptions, LightingGenerator
from obgeneration.generator.occupancy_generator import OccupancyGenerator,ClusterAssumptions


def _flatten_schedule(schedule: list[list[float]]) -> list[float]:
    return [value for week in schedule for value in week]


def _daily_slices_from_annual_schedule(annual_schedule: list[float], bins_per_day: int) -> list[list[float]]:
    assert len(annual_schedule) == 365 * bins_per_day
    return [
        annual_schedule[start:start + bins_per_day]
        for start in range(0, len(annual_schedule), bins_per_day)
    ]


def _assert_design_day_matches_extreme(
    annual_schedule: list[float],
    design_day: list[float] | None,
    mode: str,
    bins_per_day: int,
) -> None:
    daily_schedules = _daily_slices_from_annual_schedule(annual_schedule, bins_per_day)
    assert design_day is not None
    assert len(design_day) == bins_per_day
    assert any(day == design_day for day in daily_schedules)

    daily_averages = [sum(day) / len(day) for day in daily_schedules]
    design_day_average = sum(design_day) / len(design_day)

    if mode == "max":
        assert design_day_average == pytest.approx(max(daily_averages))
    elif mode == "min":
        assert design_day_average == pytest.approx(min(daily_averages))
    else:
        raise ValueError(f"Unsupported mode: {mode}")



@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def occ_1() -> Occupancy:
    occ_json ="""
        {
            "num_occupants": 3,
            "household_composition":{
                "occupants": [
                    {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                    {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                    {"weekday_cluster": "morning_away", "weekend_cluster": "afternoon_away"}
                ]
            },
            "weekday_pattern": {
                "is_always_occupied": false,
                "away_interval": {"start_hour": 8, "end_hour": 14},
                "num_of_days":3,
                "away_time_rigidness": "somewhat_variable"
            },
            "weekend_pattern": {
                "is_always_occupied": false,
                "away_interval": {"start_hour": 16, "end_hour": 20},
                "away_time_rigidness": "somewhat_variable"
            },
            "sleep_pattern": {
                "is_always_awake": false,
                "sleep_time": {"start_hour": 22, "end_hour": 6},
                "sleep_time_rigidness": "mostly_consistent"
            }
        }"""
    return Occupancy.model_validate_json(occ_json)


class TestLightingGeneration:

 
    def test_lighting_dimming(self):
        """ Test if dimming is correctly determined based on lighting usage pattern."""
        lighting1 = Lighting(
            if_led=True,
            when_daylight_bright=True
        )
        generator1 = LightingGenerator(lighting1)
        assert generator1.get_dimming() == True

        lighting2 = Lighting(
            if_led=False,
            when_daylight_bright=False
        )
        generator2 = LightingGenerator(lighting2)
        assert generator2.get_dimming() == False


    def test_lighting_house(self, occ_1, rng):
        lighting_json = """{
            "if_led": true,
            "when_daylight_bright": true
        }"""
        assumption = ClusterAssumptions.default()
        occ1 = OccupancyGenerator(occ_1, assumption, 30)
        lighting = Lighting.model_validate_json(lighting_json)
        lighting_gen = LightingGenerator(lighting)
        household_occ_sch = occ1.occupancy_annual_schedule(rng)
        _,sleep_mask = OccupancyGenerator.active_sleep_mask(household_occ_sch)
        lighting_schedule, summer_design_day, winter_design_day = lighting_gen.lighting_annual_schedule(household_occ_sch, sleep_mask)
        assert len(lighting_schedule) == 53
        for time in range(53):
            lighting_weekly = lighting_schedule[time]
            if time < 52:  # For the first 52 weeks, we have full weekly schedules
                assert len(lighting_weekly) == 24 * 7 * 2
            else:
                assert len(lighting_weekly) == 24 * 2  
            for t in range(len(lighting_weekly)):
                if sleep_mask[time][t]:
                    assert lighting_weekly[t] == 0.0
                else:
                    assert lighting_weekly[t] == 1.0

        bins_per_day = 1440 // 30
        flattened_schedule = _flatten_schedule(lighting_schedule)
        _assert_design_day_matches_extreme(flattened_schedule, summer_design_day, mode="max", bins_per_day=bins_per_day)
        _assert_design_day_matches_extreme(flattened_schedule, winter_design_day, mode="min", bins_per_day=bins_per_day)

    def test_generate_result_design_days_and_structure(self, occ_1, rng):
        lighting = Lighting(if_led=True, when_away=False, when_daylight_bright=True)
        occupancy_states = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30).occupancy_annual_schedule(rng)

        result = LightingGenerator.generate_result(
            lighting=lighting,
            occupancy_states=occupancy_states,
            lighting_assumptions=LightingAssumptions.default(),
            rng=np.random.default_rng(0),
        )

        bins_per_day = 1440 // 30
        assert len(result.annual_schedule) == 365 * bins_per_day
        assert result.summer_design_day_schedule is not None
        assert result.winter_design_day_schedule is not None
        _assert_design_day_matches_extreme(
            result.annual_schedule,
            result.summer_design_day_schedule,
            mode="max",
            bins_per_day=bins_per_day,
        )
        _assert_design_day_matches_extreme(
            result.annual_schedule,
            result.winter_design_day_schedule,
            mode="min",
            bins_per_day=bins_per_day,
        )
