import pytest

from model.lighting import Lighting
from model.occupancy import Occupancy
from generator.lighting_generator import LightingGenerator
from generator.occupancy_generator import OccupancyGenerator, OccupancyAssumptions

class TestLightingGeneration:

    @pytest.fixture
    def occ_1(self) -> Occupancy:
        occ_json = """
        {
            "num_occupants": 1,
            "household_composition": {
                "daily_commuter": 1,
                "hybrid_worker": 0,
                "stayathome": 0,
                "k12_or_daycare": 0,
                "college_student": 0
            },
            "weekday_pattern": {
                "is_always_occupied": true
            },
            "weekend_pattern": {
                "is_always_occupied": true
            },
            "sleep_time": {
                "start_hour": 22,
                "end_hour": 6
            }
        }
        """
        return Occupancy.model_validate_json(occ_json)
    
    @pytest.fixture
    def occ_2(self) -> Occupancy:
        occ_json = """
        {
            "num_occupants": 1,
            "household_composition": {
                "daily_commuter": 1,
                "hybrid_worker": 0,
                "stayathome": 0,
                "k12_or_daycare": 0,
                "college_student": 0
            },
            "weekday_pattern": {
                "is_always_occupied": false,
                "away_interval": {
                    "start_hour": 8,
                    "end_hour": 18
                },
                "num_of_days": 4
            },
            "weekend_pattern": {
                "is_always_occupied": false,
                "away_interval": {
                    "start_hour": 16,
                    "end_hour": 20
                }
            },
            "sleep_time": {
                "start_hour": 22,
                "end_hour": 6
            }
        }
        """
        return Occupancy.model_validate_json(occ_json)

    @pytest.fixture
    def occ(self, request) -> Occupancy:
        """Fixture dispatcher for parameterized tests."""
        return request.getfixturevalue(request.param)

    def test_lighting_dimming(self):
        """ Test if dimming is correctly determined based on lighting usage pattern."""
        lighting1 = Lighting(
            when_house_empty=True,
            when_daylight_bright=True
        )
        generator1 = LightingGenerator(lighting1)
        assert generator1.get_dimming() == True

        lighting2 = Lighting(
            when_house_empty=True,
            when_daylight_bright=False
        )
        generator2 = LightingGenerator(lighting2)
        assert generator2.get_dimming() == False

    @pytest.mark.parametrize("occ", ["occ_1", "occ_2"], indirect=True)
    def test_lighting_house_empty_1(self, occ):
        light_json = """
        {
            "when_house_empty": false,
            "when_daylight_bright": false
        }
        """
        assumptiion = OccupancyAssumptions.default()
        occ1 = OccupancyGenerator(occ, assumptiion)
        lighting1 = Lighting.model_validate_json(light_json)
        lighting_generator1 = LightingGenerator(lighting1)
        sleep = occ1.household_sleep_schedule()
        occupancy = occ1.household_fullweek_schedule(sleep)
        sleep_mask = occ1.get_sleep_mask(sleep,4)
        occupancy_mask = occ1.get_occupancy_mask(occupancy)
        lighting_schedule1 = lighting_generator1.lighting_weekly_schedule(occupancy_mask, sleep_mask)
        assert len(lighting_schedule1) == 24 * 7 * 4

        # Always occupied, so lighting should be always on except during sleep
        for time in range(24 * 7 * 4):
            if sleep_mask[time]:
                assert lighting_schedule1[time] == 0.0
            else:
                assert lighting_schedule1[time] == 1.0

    @pytest.mark.parametrize("occ", ["occ_1", "occ_2"], indirect=True)
    def test_lighting_house_empty_2(self, occ):
        light_json = """
        {
            "when_house_empty": true,
            "when_daylight_bright": false
        }
        """
        assumptiion = OccupancyAssumptions.default()
        occ1 = OccupancyGenerator(occ, assumptiion)
        lighting1 = Lighting.model_validate_json(light_json)
        lighting_generator1 = LightingGenerator(lighting1)
        sleep = occ1.household_sleep_schedule()
        occupancy = occ1.household_fullweek_schedule(sleep)
        sleep_mask = occ1.get_sleep_mask(sleep,4)
        occupancy_mask = occ1.get_occupancy_mask(occupancy)
        lighting_schedule1 = lighting_generator1.lighting_weekly_schedule(occupancy_mask, sleep_mask)
        assert len(lighting_schedule1) == 24 * 7 * 4

        for time in range(24 * 7 * 4):
            if sleep_mask[time]:
                assert lighting_schedule1[time] == 0.0
            elif occupancy_mask[time]:
                assert lighting_schedule1[time] == 1.0
            else:
                assert lighting_schedule1[time] == 0.0
