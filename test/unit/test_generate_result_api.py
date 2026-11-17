import json
from pathlib import Path

import numpy as np
import pytest

from obgeneration.generator.dhw_generator import DHWAssumptions, DHWGenerator
from obgeneration.generator.equipment_generator import EquipmentAssumptions, EquipmentGenerator
from obgeneration.generator.hvac_generator import HVACAssumptions, HVACGenerator
from obgeneration.generator.lighting_generator import LightingAssumptions, LightingGenerator
from obgeneration.generator.occupancy_generator import ClusterAssumptions, OccupancyGenerator
from obgeneration.generator.occupant_behavior_generator import OccupantBehaviorGenerator
from obgeneration.generator.results import HVACResult
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.generator.window_generator import WindowAssumptions, WindowGenerator
from obgeneration.model.builders import build_equipment, build_hvac, build_lighting, build_occupancy
from obgeneration.model.hvac import HVAC as HVACModel
from obgeneration.model.occupant_profile import Occupant
from obgeneration.model.window import Window, WindowOpeningBehavior
from obgeneration.stochastic.distribution import Constant
from obgeneration.model.occupancy import MobilityCluster


def _sample_inputs():
    occupancy = build_occupancy(
        mobility_clusters=[
            MobilityCluster.LONG_DAY_AWAY,
            MobilityCluster.MORNING_AWAY,
        ]
    )
    equipment = build_equipment(
        has_washer=True,
        has_dryer=True,
        has_cooking_products=True,
        has_dishwasher=True,
        num_refrigerators=1,
        washer_efficient=True,
        dryer_efficient=True,
        dishwasher_efficient=True,
        refrigerator_efficient=True,
        laundry_freq_per_week=(2, 3),
        cooking_freq_per_week=(10, 14),
        dishwasher_freq_per_week=(4, 6),
    )
    lighting = build_lighting(if_led=True, when_away=False, when_daylight_bright=True)
    hvac = build_hvac(
        has_heating=True,
        has_cooling=True,
        heating_setpoint=21.0,
        heating_setpoint_sleep=19.0,
        heating_setpoint_absent=17.0,
        cooling_setpoint=25.0,
        cooling_setpoint_sleep=27.0,
        cooling_setpoint_absent=29.0,
    )
    return occupancy, equipment, lighting, hvac


def test_occupancy_generate_result_matches_defaults():
    occupancy, _, _, _ = _sample_inputs()

    result = OccupancyGenerator.generate_result(
        occupancy=occupancy,
        cluster_assumptions=ClusterAssumptions.default(),
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )
    default_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    assert result.model_dump() == default_result.model_dump()


def test_equipment_generate_result_matches_defaults():
    occupancy, equipment, _, _ = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    result = EquipmentGenerator.generate_result(
        equipment=equipment,
        occupancy_states=occupancy_result.occupancy_states,
        num_occupants=occupancy.num_occupants,
        resolution_mins=30,
        equipment_assumptions=EquipmentAssumptions.default(30),
        rng=np.random.default_rng(42),
    )
    default_result = EquipmentGenerator.generate_with_defaults(
        equipment=equipment,
        occupancy_states=occupancy_result.occupancy_states,
        num_occupants=occupancy.num_occupants,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    assert result.model_dump() == default_result.model_dump()


def test_lighting_generate_result_matches_defaults():
    occupancy, _, lighting, _ = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    result = LightingGenerator.generate_result(
        lighting=lighting,
        occupancy_states=occupancy_result.occupancy_states,
        lighting_assumptions=LightingAssumptions.default(),
        rng=np.random.default_rng(42),
    )
    default_result = LightingGenerator.generate_with_defaults(
        lighting=lighting,
        occupancy_states=occupancy_result.occupancy_states,
        rng=np.random.default_rng(42),
    )

    assert result.model_dump() == default_result.model_dump()


def test_dhw_generate_result_matches_defaults():
    occupancy, equipment, _, _ = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )
    equipment_result = EquipmentGenerator.generate_with_defaults(
        equipment=equipment,
        occupancy_states=occupancy_result.occupancy_states,
        num_occupants=occupancy.num_occupants,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    result = DHWGenerator.generate_result(
        num_occupants=occupancy.num_occupants,
        equipment=equipment,
        occupancy_states=occupancy_result.occupancy_states,
        laundry_event_schedule=equipment_result.laundry_dhw_event_schedule,
        dishwasher_event_schedule=equipment_result.dishwasher_dhw_event_schedule,
        resolution_mins=30,
        dhw_assumptions=DHWAssumptions.default(),
    )
    default_result = DHWGenerator.generate_with_defaults(
        num_occupants=occupancy.num_occupants,
        equipment=equipment,
        occupancy_states=occupancy_result.occupancy_states,
        laundry_event_schedule=equipment_result.laundry_dhw_event_schedule,
        dishwasher_event_schedule=equipment_result.dishwasher_dhw_event_schedule,
        resolution_mins=30,
    )

    assert result.model_dump() == default_result.model_dump()


def test_hvac_generate_result_matches_defaults():
    occupancy, _, _, hvac = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    result = HVACGenerator.generate_result(
        hvac=hvac,
        occupancy_states=occupancy_result.occupancy_states,
        assumptions=HVACAssumptions.default(),
    )
    default_result = HVACGenerator.generate_with_defaults(
        hvac=hvac,
        occupancy_states=occupancy_result.occupancy_states,
    )

    assert result.model_dump() == default_result.model_dump()


def test_hvac_component_generate_result_matches_defaults():
    occupancy, _, _, hvac = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    heating = HVACGenerator.generate_heating_result(
        hvac=hvac,
        occupancy_states=occupancy_result.occupancy_states,
        assumptions=HVACAssumptions.default(),
    )
    heating_default = HVACGenerator.generate_heating_with_defaults(
        hvac=hvac,
        occupancy_states=occupancy_result.occupancy_states,
    )
    cooling = HVACGenerator.generate_cooling_result(
        hvac=hvac,
        occupancy_states=occupancy_result.occupancy_states,
        assumptions=HVACAssumptions.default(),
    )
    cooling_default = HVACGenerator.generate_cooling_with_defaults(
        hvac=hvac,
        occupancy_states=occupancy_result.occupancy_states,
    )

    assert heating is not None
    assert cooling is not None
    assert heating.model_dump() == heating_default.model_dump()
    assert cooling.model_dump() == cooling_default.model_dump()


def test_lighting_generate_result_uses_explicit_led_assumptions():
    occupancy, _, _, _ = _sample_inputs()
    lighting = build_lighting(if_led=True, when_away=False, when_daylight_bright=True)
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )
    assumptions = LightingAssumptions(
        watts_per_m2_led=Constant(3.25),
        watts_per_m2_non_led=Constant(9.75),
    )

    result = LightingGenerator.generate_result(
        lighting=lighting,
        occupancy_states=occupancy_result.occupancy_states,
        lighting_assumptions=assumptions,
        rng=np.random.default_rng(42),
    )

    assert result.peak_value == 3.25
    assert result.peak_units == "W/m2"
    assert set(result.annual_schedule).issubset({0.0, 1.0})


def test_lighting_generate_result_uses_explicit_non_led_assumptions():
    occupancy, _, _, _ = _sample_inputs()
    lighting = build_lighting(if_led=False, when_away=False, when_daylight_bright=False)
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )
    assumptions = LightingAssumptions(
        watts_per_m2_led=Constant(3.25),
        watts_per_m2_non_led=Constant(9.75),
    )

    result = LightingGenerator.generate_result(
        lighting=lighting,
        occupancy_states=occupancy_result.occupancy_states,
        lighting_assumptions=assumptions,
        rng=np.random.default_rng(42),
    )

    assert result.peak_value == 9.75
    assert result.dimming_enabled is False


def test_dhw_generate_result_uses_explicit_assumptions():
    equipment = build_equipment(
        has_washer=True,
        has_dryer=False,
        has_cooking_products=False,
        has_dishwasher=True,
        num_refrigerators=0,
        washer_efficient=True,
        dishwasher_efficient=False,
        refrigerator_efficient=None,
        laundry_freq_per_week=(1, 1),
        dishwasher_freq_per_week=(1, 1),
        cooking_freq_per_week=None,
    )
    assumptions = DHWAssumptions(
        hot_water_per_person_per_day=10.0,
        efficient_washer_per_cycle=5.0,
        inefficient_washer_per_cycle=8.0,
        efficient_dishwasher_per_cycle=7.0,
        inefficient_dishwasher_per_cycle=11.0,
    )
    weekly_occupancy_states = [[
        HouseholdOccupancyFractions(home=1.0, sleep=0.0) for _ in range(7 * 24)
    ] for _ in range(53)]
    laundry_event_schedule = [[0] * (7 * 24) for _ in range(53)]
    dishwasher_event_schedule = [[0] * (7 * 24) for _ in range(53)]
    dishwasher_event_schedule[0][24] = 1

    result = DHWGenerator.generate_result(
        num_occupants=2,
        equipment=equipment,
        occupancy_states=weekly_occupancy_states,
        laundry_event_schedule=laundry_event_schedule,
        dishwasher_event_schedule=dishwasher_event_schedule,
        resolution_mins=60,
        dhw_assumptions=assumptions,
    )

    expected_peak_bin_liters = (2 * 10.0) / 24.0 + 11.0
    expected_peak_m3_s = expected_peak_bin_liters / 1000.0 / 3600.0

    assert result.peak_value == expected_peak_m3_s
    assert max(result.annual_schedule) == 1.0
    assert min(result.annual_schedule) >= 0.0


def test_hvac_generate_result_returns_none_for_missing_systems():
    occupancy, _, _, _ = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    result = HVACGenerator.generate_result(
        hvac=HVACModel(heating=None, cooling=None),
        occupancy_states=occupancy_result.occupancy_states,
        assumptions=HVACAssumptions.default(),
    )

    assert result.heating is None
    assert result.cooling is None
    assert HVACGenerator.generate_heating_result(
        hvac=HVACModel(heating=None, cooling=None),
        occupancy_states=occupancy_result.occupancy_states,
        assumptions=HVACAssumptions.default(),
    ) is None
    assert HVACGenerator.generate_cooling_result(
        hvac=HVACModel(heating=None, cooling=None),
        occupancy_states=occupancy_result.occupancy_states,
        assumptions=HVACAssumptions.default(),
    ) is None


def test_equipment_generate_result_wrapper_keeps_normalized_schedule():
    occupancy, equipment, _, _ = _sample_inputs()
    occupancy_result = OccupancyGenerator.generate_with_defaults(
        occupancy=occupancy,
        resolution_mins=30,
        rng=np.random.default_rng(42),
    )

    result = EquipmentGenerator.generate_result(
        equipment=equipment,
        occupancy_states=occupancy_result.occupancy_states,
        num_occupants=occupancy.num_occupants,
        resolution_mins=30,
        equipment_assumptions=EquipmentAssumptions.default(30),
        rng=np.random.default_rng(42),
    )

    assert result.peak_units == "W"
    assert len(result.annual_schedule) > 0
    assert max(result.annual_schedule) == 1.0
    assert min(result.annual_schedule) >= 0.0
    assert len(result.laundry_cycles) == 53
    assert len(result.dishwasher_cycles) == 53


def _sample_profile() -> Occupant:
    occupancy, equipment, lighting, hvac = _sample_inputs()
    return Occupant(
        occupancy=occupancy,
        lighting=lighting,
        equipment=equipment,
        hvac=hvac,
        window=Window(),
    )


def test_profile_generate_with_defaults_matches_end_use_chain():
    """Same seed and order as StochasticScheduleGenerator.generate_schedules."""
    resolution_mins = 30
    profile = _sample_profile()
    chained_rng = np.random.default_rng(7)

    occupancy = OccupancyGenerator.generate_with_defaults(
        occupancy=profile.occupancy,
        resolution_mins=resolution_mins,
        rng=chained_rng,
    )
    hvac = HVACGenerator.generate_with_defaults(
        hvac=profile.hvac,
        occupancy_states=occupancy.occupancy_states,
        active_threshold=0.51,
        away_threshold=0.01,
    )
    lighting = LightingGenerator.generate_with_defaults(
        lighting=profile.lighting,
        occupancy_states=occupancy.occupancy_states,
        rng=chained_rng,
        active_threshold=0.51,
        away_threshold=0.01,
    )
    equipment = EquipmentGenerator.generate_with_defaults(
        equipment=profile.equipment,
        occupancy_states=occupancy.occupancy_states,
        num_occupants=profile.occupancy.num_occupants,
        resolution_mins=resolution_mins,
        rng=chained_rng,
    )
    dhw = DHWGenerator.generate_with_defaults(
        profile.occupancy.num_occupants,
        profile.equipment,
        occupancy.occupancy_states,
        equipment.laundry_dhw_event_schedule,
        equipment.dishwasher_dhw_event_schedule,
        resolution_mins,
    )

    result = OccupantBehaviorGenerator.generate_with_defaults(
        _sample_profile(),
        resolution_mins,
        rng=7,
    )

    assert result.occupancy.model_dump() == occupancy.model_dump()
    assert result.lighting.model_dump() == lighting.model_dump()
    assert result.equipment.model_dump() == equipment.model_dump()
    assert result.dhw.model_dump() == dhw.model_dump()
    assert result.hvac.model_dump() == hvac.model_dump()
    assert result.equipment.peak_units == "W"
    assert result.dhw.peak_units == "m3/s"
    assert result.lighting.peak_units == "W/m2"
    assert result.window is None


def test_profile_json_omits_window():
    """Sample profiles carry no window input. The loader fills an empty Window."""
    input_dir = Path(__file__).parent / "input"
    for name in ("OB_1.json", "OB_2.json", "OB_3.json"):
        path = input_dir / name
        raw = json.loads(path.read_text())
        assert "window" not in raw
        occupant = Occupant.from_json_file(path)
        assert occupant.window.heating_season is None
        assert occupant.window.cooling_season is None
        assert occupant.window.shoulder_season is None


def test_profile_json_requires_occupancy_and_lighting(tmp_path):
    lighting_only = tmp_path / "no_occupancy.json"
    lighting_only.write_text(json.dumps({"lighting": {"if_led": True}}), encoding="utf-8")
    with pytest.raises(KeyError, match="occupancy"):
        Occupant.from_json_file(lighting_only)

    occupancy_only = tmp_path / "no_lighting.json"
    occupancy_only.write_text(
        json.dumps({
            "occupancy": {
                "num_occupants": 1,
                "household_composition": {
                    "occupants": [{
                        "weekday_cluster": "mostly_home",
                        "weekend_cluster": "mostly_home",
                    }]
                },
            }
        }),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="lighting"):
        Occupant.from_json_file(occupancy_only)


def test_window_generate_result_returns_none():
    """The window call is wired through, and the schedule is still unspecified."""
    window = Window(
        heating_season=WindowOpeningBehavior.NATURAL_VENTILATION,
        cooling_season=WindowOpeningBehavior.OCCASIONALLY_OPEN,
        shoulder_season=WindowOpeningBehavior.LONG_OPEN,
    )
    occupancy_states = [[HouseholdOccupancyFractions(home=1.0, sleep=0.0)]]

    result = WindowGenerator.generate_result(
        window=window,
        occupancy_states=occupancy_states,
        assumptions=WindowAssumptions.default(),
        resolution_mins=15,
        rng=0,
        hvac=HVACResult(),
    )
    default_result = WindowGenerator.generate_with_defaults(
        window=window,
        occupancy_states=occupancy_states,
        resolution_mins=15,
        rng=0,
        hvac=HVACResult(),
    )

    assert result is None
    assert default_result is None
    assert WindowGenerator(window, WindowAssumptions.default(), 15).window_annual_schedule(
        [[True, False]]
    ) is None
    assert WindowGenerator.in_week_range(42, 42, 19)
    assert WindowGenerator.in_week_range(19, 42, 19)
    assert not WindowGenerator.in_week_range(30, 42, 19)
    assert WindowGenerator.in_week_range(22, 20, 24)
    assert not WindowGenerator.in_week_range(19, 20, 24)
