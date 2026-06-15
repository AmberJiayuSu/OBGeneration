import numpy as np
import pytest

from obgeneration.generator.occupancy_generator import OccupancyGenerator, ClusterAssumptions
from obgeneration.generator.equipment_generator import EquipmentGenerator, EquipmentAssumptions
from obgeneration.model.equipment import Equipment
from obgeneration.model.occupancy import Occupancy
from obgeneration.stochastic.markov import ClusterAssumptions as MarkovClusterAssumptions


SIM_RES = 15


@pytest.fixture
def occupancy_profile() -> Occupancy:
    occ_json = """
    {
        "num_occupants": 3,
        "household_composition": {
            "occupants": [
                {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                {"weekday_cluster": "morning_away", "weekend_cluster": "afternoon_away"}
            ]
        },
        "weekday_pattern": {
            "is_always_occupied": false,
            "away_interval": {"start_hour": 8, "end_hour": 14},
            "num_of_days": 3,
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
    }
    """
    return Occupancy.model_validate_json(occ_json)


@pytest.fixture
def equipment_profile() -> Equipment:
    equipment_json = """
    {
        "laundry": {
            "has_washer": true,
            "washer_efficient": true,
            "has_dryer": true,
            "dryer_efficient": false,
            "usage_frequency_per_week": {"min": 2, "max": 4}
        },
        "refrigerator": {
            "has_refrigerator": true,
            "efficient_refrigerator": true,
            "number_of_refrigerators": 1
        },
        "dishwasher": {
            "has_dishwasher": true,
            "dishwasher_efficient": true,
            "dishwashing_operational_logic": {
                "pattern_type": "daily_batch_if_cooked"
            }
        },
        "cooking_products": {
            "has_cooking_products": true,
            "cooking_products_fuel": "electric",
            "usage_frequency_per_week": {"min": 10, "max": 15}
        }
    }
    """
    return Equipment.model_validate_json(equipment_json)


class TestRngReproducibility:
    def test_cluster_sampling_is_reproducible_with_same_seed(self):
        assumptions = MarkovClusterAssumptions.default()

        weeks_1 = assumptions.sample_cluster_annually(0, 0, SIM_RES, np.random.default_rng(1234))
        weeks_2 = assumptions.sample_cluster_annually(0, 0, SIM_RES, np.random.default_rng(1234))

        assert weeks_1 == weeks_2

    def test_occupancy_generation_is_reproducible_with_same_seed(self, occupancy_profile):
        assumptions = ClusterAssumptions.default()

        gen_1 = OccupancyGenerator(occupancy_profile, assumptions, SIM_RES)
        gen_2 = OccupancyGenerator(occupancy_profile, assumptions, SIM_RES)

        schedule_1 = gen_1.occupancy_annual_schedule(np.random.default_rng(2024))
        schedule_2 = gen_2.occupancy_annual_schedule(np.random.default_rng(2024))

        assert schedule_1 == schedule_2

    def test_equipment_generation_is_reproducible_with_same_seed(self, occupancy_profile, equipment_profile):
        occupancy_states = OccupancyGenerator(
            occupancy_profile,
            ClusterAssumptions.default(),
            SIM_RES,
        ).occupancy_annual_schedule(np.random.default_rng(11))

        assumptions = EquipmentAssumptions.default(SIM_RES)
        gen_1 = EquipmentGenerator(
            equipment=equipment_profile,
            occupancy_state=occupancy_states,
            num_occupants=occupancy_profile.num_occupants,
            resolution_mins=SIM_RES,
            equipment_assumptions=assumptions,
        )
        gen_2 = EquipmentGenerator(
            equipment=equipment_profile,
            occupancy_state=occupancy_states,
            num_occupants=occupancy_profile.num_occupants,
            resolution_mins=SIM_RES,
            equipment_assumptions=assumptions,
        )

        annual_1 = gen_1.equipment_annual_schedule(np.random.default_rng(99))
        annual_2 = gen_2.equipment_annual_schedule(np.random.default_rng(99))

        assert annual_1 == annual_2

    def test_reusing_advanced_rng_changes_output(self, occupancy_profile):
        assumptions = ClusterAssumptions.default()
        generator = OccupancyGenerator(occupancy_profile, assumptions, SIM_RES)
        rng = np.random.default_rng(7)

        schedule_1 = generator.occupancy_annual_schedule(rng)
        schedule_2 = generator.occupancy_annual_schedule(rng)

        assert schedule_1 != schedule_2
