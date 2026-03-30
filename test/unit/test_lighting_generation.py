import pytest

from ob_generation.model.lighting import Lighting
from ob_generation.model.occupancy import Occupancy
from ob_generation.generator.lighting_generator import LightingGenerator
from ob_generation.generator.occupancy_generator import OccupancyGenerator,ClusterAssumptions



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
            when_daylight_bright=True
        )
        generator1 = LightingGenerator(lighting1)
        assert generator1.get_dimming() == True

        lighting2 = Lighting(
            when_daylight_bright=False
        )
        generator2 = LightingGenerator(lighting2)
        assert generator2.get_dimming() == False


    def test_lighting_house(self, occ_1):
        lighting_json = """{
            "when_daylight_bright": true
        }"""
        assumption = ClusterAssumptions.default()
        occ1 = OccupancyGenerator(occ_1, assumption, 30)
        lighting = Lighting.model_validate_json(lighting_json)
        lighting_gen = LightingGenerator(lighting)
        household_occ_sch = occ1.generate()
        _,sleep_mask = OccupancyGenerator.active_sleep_mask(household_occ_sch)
        lighting_schedule = lighting_gen.lighting_annual_schedule(sleep_mask)
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
