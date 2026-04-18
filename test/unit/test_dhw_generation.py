from pathlib import Path

import pytest

from obgeneration.generator.dhw_generator import DHWAssumptions, DHWGenerator
from obgeneration.model.equipment import Equipment


SIM_RES = 15


@pytest.fixture
def empty_cycles() -> tuple[list[list[int]], list[list[int]]]:
    laundry = [[0, 0, 0, 0, 0, 0, 0] for _ in range(53)]
    dishwasher = [[0, 0, 0, 0, 0, 0, 0] for _ in range(53)]
    return laundry, dishwasher


@pytest.fixture
def no_hot_water_equipment() -> Equipment:
    equipment_json = """
    {
        "laundry": {
            "has_washer": false,
            "has_dryer": false
        },
        "dishwasher": {
            "has_dishwasher": false
        },
        "refrigerator": {
            "has_refrigerator": false
        },
        "cooking_products": {
            "has_cooking_products": false
        }
    }
    """
    return Equipment.model_validate_json(equipment_json)


@pytest.fixture
def efficient_hot_water_equipment() -> Equipment:
    equipment_json = """
    {
        "laundry": {
            "has_washer": true,
            "washer_efficient": true,
            "has_dryer": false
        },
        "dishwasher": {
            "has_dishwasher": true,
            "dishwasher_efficient": true
        }
    }
    """
    return Equipment.model_validate_json(equipment_json)


@pytest.fixture
def inefficient_hot_water_equipment() -> Equipment:
    equipment_json = """
    {
        "laundry": {
            "has_washer": true,
            "washer_efficient": false,
            "has_dryer": false
        },
        "dishwasher": {
            "has_dishwasher": true,
            "dishwasher_efficient": false
        }
    }
    """
    return Equipment.model_validate_json(equipment_json)


class TestDHWAssumptions:
    def test_default_values_are_positive(self):
        assumptions = DHWAssumptions.default()

        assert assumptions.hot_water_per_person_per_day > 0
        assert assumptions.efficient_washer_per_cycle > 0
        assert assumptions.inefficient_washer_per_cycle > assumptions.efficient_washer_per_cycle
        assert assumptions.efficient_dishwasher_per_cycle > 0
        assert assumptions.inefficient_dishwasher_per_cycle > assumptions.efficient_dishwasher_per_cycle

    def test_from_json_file(self, tmp_path: Path):
        assumptions_path = tmp_path / "dhw_assumptions.json"
        assumptions_path.write_text(
            """
            {
                "hot_water_per_person_per_day": 50.0,
                "efficient_washer_per_cycle": 5.0,
                "inefficient_washer_per_cycle": 12.0,
                "efficient_dishwasher_per_cycle": 10.0,
                "inefficient_dishwasher_per_cycle": 25.0
            }
            """
        )

        assumptions = DHWAssumptions.from_json_file(assumptions_path)

        assert assumptions.hot_water_per_person_per_day == 50.0
        assert assumptions.efficient_washer_per_cycle == 5.0
        assert assumptions.inefficient_washer_per_cycle == 12.0
        assert assumptions.efficient_dishwasher_per_cycle == 10.0
        assert assumptions.inefficient_dishwasher_per_cycle == 25.0


class TestDHWGenerator:
    def test_occupants_only_weekly_dhw(self, no_hot_water_equipment, empty_cycles):
        laundry_cycles, dishwasher_cycles = empty_cycles
        assumptions = DHWAssumptions.default()
        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=no_hot_water_equipment,
            num_occupants=3,
            laundry_cycles_per_day=laundry_cycles,
            dishwasher_cycles_per_day=dishwasher_cycles,
            resolution_mins=SIM_RES,
        )

        weekly = generator.dhw_weekly_schedule(laundry_cycles[0], dishwasher_cycles[0])
        expected_daily = 3 * assumptions.hot_water_per_person_per_day

        assert len(weekly) == 7
        assert weekly == pytest.approx([expected_daily] * 7)

    def test_weekly_laundry_dhw_uses_efficiency(self, efficient_hot_water_equipment, inefficient_hot_water_equipment):
        assumptions = DHWAssumptions.default()
        weekly_cycles = [0, 1, 2, 0, 3, 1, 0]

        efficient = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=efficient_hot_water_equipment,
            num_occupants=1,
            laundry_cycles_per_day=[weekly_cycles],
            dishwasher_cycles_per_day=[[0] * 7],
            resolution_mins=SIM_RES,
        )
        inefficient = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=inefficient_hot_water_equipment,
            num_occupants=1,
            laundry_cycles_per_day=[weekly_cycles],
            dishwasher_cycles_per_day=[[0] * 7],
            resolution_mins=SIM_RES,
        )

        efficient_daily = efficient.weekly_laundry_dhw(weekly_cycles)
        inefficient_daily = inefficient.weekly_laundry_dhw(weekly_cycles)

        assert efficient_daily == pytest.approx([c * assumptions.efficient_washer_per_cycle for c in weekly_cycles])
        assert inefficient_daily == pytest.approx([c * assumptions.inefficient_washer_per_cycle for c in weekly_cycles])

    def test_weekly_dishwasher_dhw_uses_efficiency(self, efficient_hot_water_equipment, inefficient_hot_water_equipment):
        assumptions = DHWAssumptions.default()
        weekly_cycles = [1, 0, 1, 2, 0, 1, 3]

        efficient = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=efficient_hot_water_equipment,
            num_occupants=1,
            laundry_cycles_per_day=[[0] * 7],
            dishwasher_cycles_per_day=[weekly_cycles],
            resolution_mins=SIM_RES,
        )
        inefficient = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=inefficient_hot_water_equipment,
            num_occupants=1,
            laundry_cycles_per_day=[[0] * 7],
            dishwasher_cycles_per_day=[weekly_cycles],
            resolution_mins=SIM_RES,
        )

        efficient_daily = efficient.weekly_dishwasher_dhw(weekly_cycles)
        inefficient_daily = inefficient.weekly_dishwasher_dhw(weekly_cycles)

        assert efficient_daily == pytest.approx([c * assumptions.efficient_dishwasher_per_cycle for c in weekly_cycles])
        assert inefficient_daily == pytest.approx([c * assumptions.inefficient_dishwasher_per_cycle for c in weekly_cycles])

    def test_dhw_annual_schedule_structure_and_normalization(self, efficient_hot_water_equipment):
        assumptions = DHWAssumptions.default()
        laundry_cycles = [[1, 0, 0, 1, 0, 0, 0] for _ in range(53)]
        dishwasher_cycles = [[0, 1, 0, 0, 1, 0, 0] for _ in range(53)]

        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=efficient_hot_water_equipment,
            num_occupants=2,
            laundry_cycles_per_day=laundry_cycles,
            dishwasher_cycles_per_day=dishwasher_cycles,
            resolution_mins=SIM_RES,
        )

        max_flow_rate, annual_schedule = generator.dhw_annual_schedule()

        assert max_flow_rate > 0.0
        assert len(annual_schedule) == 53
        assert all(len(week) == 7 * 24 * (60 // SIM_RES) for week in annual_schedule)
        assert max(max(week) for week in annual_schedule) == pytest.approx(1.0)
        assert min(min(week) for week in annual_schedule) >= 0.0

    def test_generate_with_defaults_matches_direct_construction(self, efficient_hot_water_equipment):
        laundry_cycles = [[1, 0, 0, 1, 0, 0, 0] for _ in range(53)]
        dishwasher_cycles = [[0, 1, 0, 0, 1, 0, 0] for _ in range(53)]

        direct = DHWGenerator(
            dhw_assumptions=DHWAssumptions.default(),
            equipment=efficient_hot_water_equipment,
            num_occupants=2,
            laundry_cycles_per_day=laundry_cycles,
            dishwasher_cycles_per_day=dishwasher_cycles,
            resolution_mins=SIM_RES,
        ).dhw_annual_schedule()

        generated = DHWGenerator.generate_with_defaults(
            num_occupants=2,
            equipment=efficient_hot_water_equipment,
            laundry_cycles_per_day=laundry_cycles,
            dishwasher_cycles_per_day=dishwasher_cycles,
            resolution_mins=SIM_RES,
        )

        assert generated == direct
