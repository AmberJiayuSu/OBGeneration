from pathlib import Path

import pytest

from obgeneration.generator.dhw_generator import (
    DHWAssumptions,
    DHWFlatGenerator,
    DHWGenerator,
)
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.model.equipment import Equipment


SIM_RES = 15
WEEKS_PER_YEAR = 53


def _daily_slices_from_schedule_prefix(
    annual_schedule: list[float],
    bins_per_day: int,
    num_days: int,
) -> list[list[float]]:
    return [
        annual_schedule[start : start + bins_per_day]
        for start in range(0, num_days * bins_per_day, bins_per_day)
    ]


def _assert_design_day_matches_max(
    annual_schedule: list[float],
    design_day: list[float] | None,
    bins_per_day: int,
    num_days: int,
) -> None:
    daily_schedules = _daily_slices_from_schedule_prefix(
        annual_schedule,
        bins_per_day,
        num_days,
    )
    assert design_day is not None
    assert len(design_day) == bins_per_day
    assert any(day == design_day for day in daily_schedules)

    daily_averages = [sum(day) / len(day) for day in daily_schedules]
    design_day_average = sum(design_day) / len(design_day)
    assert design_day_average == pytest.approx(max(daily_averages))


def _make_weekly_states(day_home_weights: list[float]) -> list[HouseholdOccupancyFractions]:
    return [
        HouseholdOccupancyFractions(home=home, sleep=0.0)
        for _ in range(7)
        for home in day_home_weights
    ]


def _repeat_week(week: list[int] | list[float] | list[HouseholdOccupancyFractions]) -> list[list]:
    return [list(week) for _ in range(WEEKS_PER_YEAR)]


@pytest.fixture
def bins_per_day() -> int:
    return 24 * (60 // SIM_RES)


@pytest.fixture
def bins_per_week(bins_per_day: int) -> int:
    return 7 * bins_per_day


@pytest.fixture
def empty_daily_cycles() -> tuple[list[list[int]], list[list[int]]]:
    laundry = [[0, 0, 0, 0, 0, 0, 0] for _ in range(WEEKS_PER_YEAR)]
    dishwasher = [[0, 0, 0, 0, 0, 0, 0] for _ in range(WEEKS_PER_YEAR)]
    return laundry, dishwasher


@pytest.fixture
def empty_event_schedules(bins_per_week: int) -> tuple[list[list[int]], list[list[int]]]:
    laundry = [[0] * bins_per_week for _ in range(WEEKS_PER_YEAR)]
    dishwasher = [[0] * bins_per_week for _ in range(WEEKS_PER_YEAR)]
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


class TestDHWFlatGenerator:
    def test_occupants_only_weekly_dhw(self, no_hot_water_equipment, empty_daily_cycles):
        laundry_cycles, dishwasher_cycles = empty_daily_cycles
        assumptions = DHWAssumptions.default()
        generator = DHWFlatGenerator(
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

    def test_appliance_daily_liters_follow_efficiency(
        self,
        efficient_hot_water_equipment,
        inefficient_hot_water_equipment,
    ):
        assumptions = DHWAssumptions.default()
        laundry_cycles = [0, 1, 2, 0, 3, 1, 0]
        dishwasher_cycles = [1, 0, 1, 2, 0, 1, 3]

        efficient = DHWFlatGenerator(
            dhw_assumptions=assumptions,
            equipment=efficient_hot_water_equipment,
            num_occupants=0,
            laundry_cycles_per_day=[laundry_cycles],
            dishwasher_cycles_per_day=[dishwasher_cycles],
            resolution_mins=SIM_RES,
        )
        inefficient = DHWFlatGenerator(
            dhw_assumptions=assumptions,
            equipment=inefficient_hot_water_equipment,
            num_occupants=0,
            laundry_cycles_per_day=[laundry_cycles],
            dishwasher_cycles_per_day=[dishwasher_cycles],
            resolution_mins=SIM_RES,
        )

        assert efficient.weekly_laundry_dhw(laundry_cycles) == pytest.approx(
            [c * assumptions.efficient_washer_per_cycle for c in laundry_cycles]
        )
        assert inefficient.weekly_laundry_dhw(laundry_cycles) == pytest.approx(
            [c * assumptions.inefficient_washer_per_cycle for c in laundry_cycles]
        )
        assert efficient.weekly_dishwasher_dhw(dishwasher_cycles) == pytest.approx(
            [c * assumptions.efficient_dishwasher_per_cycle for c in dishwasher_cycles]
        )
        assert inefficient.weekly_dishwasher_dhw(dishwasher_cycles) == pytest.approx(
            [c * assumptions.inefficient_dishwasher_per_cycle for c in dishwasher_cycles]
        )


class TestDHWGenerator:
    def test_occupant_background_uses_home_ratio(
        self,
        no_hot_water_equipment,
        bins_per_day,
        empty_event_schedules,
    ):
        assumptions = DHWAssumptions.default()
        daily_total = assumptions.hot_water_per_person_per_day
        day_weights = [1.0, 3.0] + [0.0] * (bins_per_day - 2)
        weekly_states = _make_weekly_states(day_weights)
        laundry_events, dishwasher_events = empty_event_schedules

        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=no_hot_water_equipment,
            num_occupants=1,
            occupancy_states=[weekly_states],
            laundry_event_schedule=[laundry_events[0]],
            dishwasher_event_schedule=[dishwasher_events[0]],
            resolution_mins=SIM_RES,
        )

        weekly_flow = generator.dhw_weekly_flow_schedule(
            weekly_occupancy_states=weekly_states,
            weekly_laundry_event_schedule=laundry_events[0],
            weekly_dishwasher_event_schedule=dishwasher_events[0],
        )

        first_day = weekly_flow[:bins_per_day]
        assert first_day[0] == pytest.approx(daily_total * 0.25)
        assert first_day[1] == pytest.approx(daily_total * 0.75)
        assert sum(first_day[2:]) == pytest.approx(0.0)
        assert sum(first_day) == pytest.approx(daily_total)

    def test_occupant_background_falls_back_to_uniform_when_no_one_is_home(
        self,
        no_hot_water_equipment,
        bins_per_day,
        empty_event_schedules,
    ):
        assumptions = DHWAssumptions.default()
        uniform_bin_value = assumptions.hot_water_per_person_per_day / bins_per_day
        weekly_states = _make_weekly_states([0.0] * bins_per_day)
        laundry_events, dishwasher_events = empty_event_schedules

        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=no_hot_water_equipment,
            num_occupants=1,
            occupancy_states=[weekly_states],
            laundry_event_schedule=[laundry_events[0]],
            dishwasher_event_schedule=[dishwasher_events[0]],
            resolution_mins=SIM_RES,
        )

        weekly_flow = generator.dhw_weekly_flow_schedule(
            weekly_occupancy_states=weekly_states,
            weekly_laundry_event_schedule=laundry_events[0],
            weekly_dishwasher_event_schedule=dishwasher_events[0],
        )

        assert weekly_flow[:bins_per_day] == pytest.approx([uniform_bin_value] * bins_per_day)

    def test_appliance_events_are_split_evenly_across_event_bins(
        self,
        efficient_hot_water_equipment,
        bins_per_week,
    ):
        assumptions = DHWAssumptions.default()
        weekly_states = _make_weekly_states([0.0] * (bins_per_week // 7))
        laundry_events = [0] * bins_per_week
        dishwasher_events = [0] * bins_per_week
        laundry_events[10] = 1
        laundry_events[11] = 1
        dishwasher_events[25] = 2
        dishwasher_events[26] = 2
        dishwasher_events[27] = 2

        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=efficient_hot_water_equipment,
            num_occupants=0,
            occupancy_states=[weekly_states],
            laundry_event_schedule=[laundry_events],
            dishwasher_event_schedule=[dishwasher_events],
            resolution_mins=SIM_RES,
        )

        weekly_flow = generator.dhw_weekly_flow_schedule(
            weekly_occupancy_states=weekly_states,
            weekly_laundry_event_schedule=laundry_events,
            weekly_dishwasher_event_schedule=dishwasher_events,
        )

        assert weekly_flow[10] == pytest.approx(assumptions.efficient_washer_per_cycle / 2)
        assert weekly_flow[11] == pytest.approx(assumptions.efficient_washer_per_cycle / 2)
        assert weekly_flow[25] == pytest.approx(assumptions.efficient_dishwasher_per_cycle / 3)
        assert weekly_flow[26] == pytest.approx(assumptions.efficient_dishwasher_per_cycle / 3)
        assert weekly_flow[27] == pytest.approx(assumptions.efficient_dishwasher_per_cycle / 3)
        assert sum(weekly_flow) == pytest.approx(
            assumptions.efficient_washer_per_cycle + assumptions.efficient_dishwasher_per_cycle
        )

    def test_weekly_flow_schedule_validates_schedule_lengths(
        self,
        no_hot_water_equipment,
        bins_per_day,
    ):
        assumptions = DHWAssumptions.default()
        weekly_states = _make_weekly_states([1.0] * bins_per_day)
        laundry_events = [0] * (len(weekly_states) - 1)
        dishwasher_events = [0] * len(weekly_states)

        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=no_hot_water_equipment,
            num_occupants=1,
            occupancy_states=[weekly_states],
            laundry_event_schedule=[laundry_events],
            dishwasher_event_schedule=[dishwasher_events],
            resolution_mins=SIM_RES,
        )

        with pytest.raises(ValueError, match="same weekly length"):
            generator.dhw_weekly_flow_schedule(
                weekly_occupancy_states=weekly_states,
                weekly_laundry_event_schedule=laundry_events,
                weekly_dishwasher_event_schedule=dishwasher_events,
            )

    def test_dhw_annual_schedule_normalizes_by_peak_bin(
        self,
        no_hot_water_equipment,
        bins_per_day,
        bins_per_week,
    ):
        assumptions = DHWAssumptions.default()
        day_weights = [1.0] + [0.0] * (bins_per_day - 1)
        weekly_states = _make_weekly_states(day_weights)

        generator = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=no_hot_water_equipment,
            num_occupants=1,
            occupancy_states=_repeat_week(weekly_states),
            laundry_event_schedule=_repeat_week([0] * bins_per_week),
            dishwasher_event_schedule=_repeat_week([0] * bins_per_week),
            resolution_mins=SIM_RES,
        )

        peak_flow, annual_schedule = generator.dhw_annual_schedule()
        expected_peak_flow = assumptions.hot_water_per_person_per_day / 1000.0 / (SIM_RES * 60)

        assert peak_flow == pytest.approx(expected_peak_flow)
        assert len(annual_schedule) == WEEKS_PER_YEAR
        assert all(len(week) == bins_per_week for week in annual_schedule)
        assert max(max(week) for week in annual_schedule) == pytest.approx(1.0)
        assert annual_schedule[0][0] == pytest.approx(1.0)
        assert annual_schedule[0][1] == pytest.approx(0.0)

    def test_generate_result_matches_direct_event_schedule(
        self,
        efficient_hot_water_equipment,
        bins_per_day,
        bins_per_week,
    ):
        assumptions = DHWAssumptions.default()
        day_weights = [1.0] + [0.0] * (bins_per_day - 1)
        weekly_states = _make_weekly_states(day_weights)
        laundry_week = [0] * bins_per_week
        dishwasher_week = [0] * bins_per_week
        laundry_week[0] = 1
        laundry_week[1] = 1
        dishwasher_week[bins_per_day] = 1
        dishwasher_week[bins_per_day + 1] = 1
        dishwasher_week[bins_per_day + 2] = 1

        direct_peak, direct_schedule = DHWGenerator(
            dhw_assumptions=assumptions,
            equipment=efficient_hot_water_equipment,
            num_occupants=1,
            occupancy_states=_repeat_week(weekly_states),
            laundry_event_schedule=_repeat_week(laundry_week),
            dishwasher_event_schedule=_repeat_week(dishwasher_week),
            resolution_mins=SIM_RES,
        ).dhw_annual_schedule()
        flattened_direct_schedule = [value for week in direct_schedule for value in week]

        result = DHWGenerator.generate_result(
            num_occupants=1,
            equipment=efficient_hot_water_equipment,
            occupancy_states=_repeat_week(weekly_states),
            laundry_event_schedule=_repeat_week(laundry_week),
            dishwasher_event_schedule=_repeat_week(dishwasher_week),
            resolution_mins=SIM_RES,
            dhw_assumptions=assumptions,
        )

        assert result.peak_units == "m3/s"
        assert result.peak_value == pytest.approx(direct_peak)
        assert list(result.annual_schedule) == pytest.approx(flattened_direct_schedule)
        assert len(result.annual_schedule) == WEEKS_PER_YEAR * bins_per_week
        assert result.summer_design_day_schedule == result.winter_design_day_schedule
        _assert_design_day_matches_max(
            result.annual_schedule,
            result.summer_design_day_schedule,
            bins_per_day=bins_per_day,
            num_days=365,
        )
