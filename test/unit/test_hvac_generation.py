import pytest
import numpy as np

from obgeneration.generator.occupancy_generator import OccupancyGenerator, ClusterAssumptions
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.model.hvac import HVAC
from obgeneration.generator.hvac_generator import HVACAssumptions, HVACGenerator
from obgeneration.model.occupancy import Occupancy


def _flatten_schedule(schedule: list[list[float]]) -> list[float]:
    return [value for week in schedule for value in week]


def _daily_slices(schedule: list[list[float]]) -> list[list[float]]:
    flat_schedule = _flatten_schedule(schedule)
    timesteps_per_day = len(flat_schedule) // 365
    assert timesteps_per_day * 365 == len(flat_schedule)
    return [
        flat_schedule[start:start + timesteps_per_day]
        for start in range(0, len(flat_schedule), timesteps_per_day)
    ]


def _assert_schedule_structure(
    schedule: list[list[float]] | None,
    reference_mask: list[list[bool]],
) -> None:
    assert schedule is not None
    assert len(schedule) == len(reference_mask)
    assert all(len(schedule[w]) == len(reference_mask[w]) for w in range(len(schedule)))


def _assert_design_day_matches_extreme(
    schedule: list[list[float]],
    design_day: list[float] | None,
    mode: str,
) -> None:
    daily_schedules = _daily_slices(schedule)
    assert design_day is not None
    assert len(design_day) == len(daily_schedules[0])
    assert any(day == design_day for day in daily_schedules)

    daily_averages = [sum(day) / len(day) for day in daily_schedules]
    design_day_avg = sum(design_day) / len(design_day)

    if mode == "max":
        assert design_day_avg == pytest.approx(max(daily_averages))
    elif mode == "min":
        assert design_day_avg == pytest.approx(min(daily_averages))
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


class TestHVACGeneration:

    @staticmethod
    def _annual_occupancy_with_single_active_spike(
        spike_index: int,
        bins_per_day: int = 96,
    ) -> list[list[HouseholdOccupancyFractions]]:
        week_len = 7 * bins_per_day
        asleep = HouseholdOccupancyFractions(home=0.0, sleep=1.0)
        active = HouseholdOccupancyFractions(home=1.0, sleep=0.0)

        first_week = [asleep] * week_len
        first_week[spike_index] = active
        return [first_week] + [[asleep] * week_len for _ in range(51)] + [[asleep] * bins_per_day]

    def test_smooth_hvac_state_sequence_removes_short_spike(self):
        active_mask = [False, False, True, False, False]
        sleep_mask = [True, True, False, True, True]

        smoothed_active, smoothed_sleep = HVACGenerator._smooth_hvac_state_sequence(
            active_mask,
            sleep_mask,
            minimum_state_bins=2,
        )

        assert smoothed_active == [False, False, False, False, False]
        assert smoothed_sleep == [True, True, True, True, True]

    def test_generate_result_smoothing_removes_short_hvac_spike(self):
        hvac_json = """
        {
            "heating": {
                "type": "thermostat",
                "active_setpoint": 21.0,
                "sleep_setpoint": 18.0,
                "absent_setpoint": 16.0
            },
            "cooling": {
                "type": "thermostat",
                "active_setpoint": 24.0,
                "sleep_setpoint": 27.0,
                "absent_setpoint": 30.0
            }
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()
        spike_index = 100
        occupancy_states = self._annual_occupancy_with_single_active_spike(spike_index)

        baseline = HVACGenerator.generate_result(
            hvac=hvac,
            occupancy_states=occupancy_states,
            assumptions=assumptions,
            min_state_mins=None,
        )
        smoothed = HVACGenerator.generate_result(
            hvac=hvac,
            occupancy_states=occupancy_states,
            assumptions=assumptions,
            min_state_mins=60,
        )

        assert baseline.heating is not None
        assert baseline.cooling is not None
        assert smoothed.heating is not None
        assert smoothed.cooling is not None

        assert baseline.heating.annual_schedule[spike_index] == 21.0
        assert baseline.cooling.annual_schedule[spike_index] == 24.0
        assert smoothed.heating.annual_schedule[spike_index] == 18.0
        assert smoothed.cooling.annual_schedule[spike_index] == 27.0

    def test_component_generate_result_applies_smoothing_option(self):
        hvac_json = """
        {
            "heating": {
                "type": "thermostat",
                "active_setpoint": 21.0,
                "sleep_setpoint": 18.0,
                "absent_setpoint": 16.0
            },
            "cooling": {
                "type": "thermostat",
                "active_setpoint": 24.0,
                "sleep_setpoint": 27.0,
                "absent_setpoint": 30.0
            }
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()
        spike_index = 100
        occupancy_states = self._annual_occupancy_with_single_active_spike(spike_index)

        heating = HVACGenerator.generate_heating_result(
            hvac=hvac,
            occupancy_states=occupancy_states,
            assumptions=assumptions,
            min_state_mins=60,
        )
        cooling = HVACGenerator.generate_cooling_result(
            hvac=hvac,
            occupancy_states=occupancy_states,
            assumptions=assumptions,
            min_state_mins=60,
        )

        assert heating is not None
        assert cooling is not None
        assert heating.annual_schedule[spike_index] == 18.0
        assert cooling.annual_schedule[spike_index] == 27.0

    
    def test_no_hvac(self, occ_1, rng):
        """Test case where there is no HVAC system."""
        hvac_json = """
        {
            "heating": null,
            "cooling": null
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.occupancy_annual_schedule(rng)
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)


        heating, winter_design_day = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling, summer_design_day = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        assert heating is None
        assert cooling is None
        assert winter_design_day is None
        assert summer_design_day is None



    def test_heating_no_control(self, occ_1, rng):
        """Test case where there is heating but no control schedule."""
        hvac_json = """
        {
            "heating": {
                "type": "no_control"
            },
            "cooling": {
                "type": "no_control"
            }
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.occupancy_annual_schedule(rng)
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating, winter_design_day = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling, summer_design_day = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        _assert_schedule_structure(heating, active_mask)
        _assert_design_day_matches_extreme(heating, winter_design_day, mode="max")
        if_same_values = all(all(h == HVACAssumptions.default().heating_defaults.active_setpoint for h in heating[w]) for w in range(len(heating)))
        assert if_same_values
        _assert_schedule_structure(cooling, active_mask)
        _assert_design_day_matches_extreme(cooling, summer_design_day, mode="min")
        if_same_values = all(all(c == HVACAssumptions.default().cooling_defaults.active_setpoint for c in cooling[w]) for w in range(len(cooling)))
        assert if_same_values


    def test_binary_control(self,occ_1, rng):
        hvac_json = """
        {
            "heating": {
                "type": "binary",
                "active_state": true,
                "sleep_state": true,
                "absent_state": false
            },
            "cooling": {
                "type": "binary",
                "active_state": true,
                "sleep_state": false,
                "absent_state": false
            }
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()

        hvac_generator = HVACGenerator(hvac, assumptions)

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.occupancy_annual_schedule(rng)
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating, winter_design_day = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling, summer_design_day = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        _assert_schedule_structure(heating, active_mask)
        _assert_design_day_matches_extreme(heating, winter_design_day, mode="max")

        sleep_on = all(all(heating[w][h] == assumptions.heating_defaults.sleep_setpoint 
                           for h in range(len(heating[w])) if sleep_mask[w][h]) 
                       for w in range(len(heating)))
        assert sleep_on

        active_on = all(all(heating[w][h] == assumptions.heating_defaults.active_setpoint 
                            for h in range(len(heating[w])) if active_mask[w][h] ) 
                        for w in range(len(heating)))
        assert active_on

        absent_off = all(all(heating[w][h] == assumptions.minimum_heating_setpoint
                             for h in range(len(heating[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_off

        _assert_schedule_structure(cooling, active_mask)
        _assert_design_day_matches_extreme(cooling, summer_design_day, mode="min")
        sleep_off = all(all(cooling[w][h] == assumptions.maximum_cooling_setpoint
                           for h in range(len(cooling[w])) if sleep_mask[w][h]) 
                       for w in range(len(cooling)))
        assert sleep_off
        active_on = all(all(cooling[w][h] == assumptions.cooling_defaults.active_setpoint 
                            for h in range(len(cooling[w])) if active_mask[w][h] ) 
                        for w in range(len(cooling)))
        assert active_on
        absent_off = all(all(cooling[w][h] == assumptions.maximum_cooling_setpoint
                             for h in range(len(cooling[w])) if not active_mask[w][h]) 
                         for w in range(len(cooling)))
        assert absent_off




    def test_heating_valve_control(self,occ_1, rng):
        hvac_json = """
        {
            "heating": {
                "type": "valve",
                "active_level": "high",
                "sleep_level": "medium",
                "absent_level": "low"
            },
            "cooling": null
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()

        hvac_generator = HVACGenerator(hvac, assumptions)

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.occupancy_annual_schedule(rng)
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating, winter_design_day = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling, summer_design_day = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        _assert_schedule_structure(heating, active_mask)
        _assert_design_day_matches_extreme(heating, winter_design_day, mode="max")

        sleep_on = all(all(heating[w][h] == assumptions.trv.valve_medium
                           for h in range(len(heating[w])) if sleep_mask[w][h]) 
                       for w in range(len(heating)))
        assert sleep_on

        active_on = all(all(heating[w][h] == assumptions.trv.valve_high
                            for h in range(len(heating[w])) if active_mask[w][h] ) 
                        for w in range(len(heating)))
        assert active_on

        absent_off = all(all(heating[w][h] == assumptions.trv.valve_low
                             for h in range(len(heating[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_off
        assert cooling is None
        assert summer_design_day is None



    def test_setpoint_control(self,occ_1, rng):
        hvac_json = """
        {
            "heating": {
                "type": "thermostat",
                "active_setpoint": 21.0,
                "sleep_setpoint": 18.0,
                "absent_setpoint": 16.0
            },
            "cooling": {
                "type": "thermostat",
                "active_setpoint": 24.0,
                "sleep_setpoint": 27.0,
                "absent_setpoint": 30.0
            }
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()

        hvac_generator = HVACGenerator(hvac, assumptions)
       
        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.occupancy_annual_schedule(rng)
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating, winter_design_day = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling, summer_design_day = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        _assert_schedule_structure(heating, active_mask)
        _assert_design_day_matches_extreme(heating, winter_design_day, mode="max")
        sleep_setpoint = all(all(heating[w][h] == 18.0
                           for h in range(len(heating[w])) if sleep_mask[w][h]) 
                       for w in range(len(heating)))
        assert sleep_setpoint
        active_setpoint = all(all(heating[w][h] == 21.0
                            for h in range(len(heating[w])) if active_mask[w][h])
                        for w in range(len(heating)))
        assert active_setpoint
        absent_setpoint = all(all(heating[w][h] == 16.0
                             for h in range(len(heating[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_setpoint

        _assert_schedule_structure(cooling, active_mask)
        _assert_design_day_matches_extreme(cooling, summer_design_day, mode="min")
        sleep_setpoint = all(all(cooling[w][h] == 27.0
                           for h in range(len(cooling[w])) if sleep_mask[w][h]) 
                       for w in range(len(cooling)))
        assert sleep_setpoint
        active_setpoint = all(all(cooling[w][h] == 24.0
                            for h in range(len(cooling[w])) if active_mask[w][h] )
                        for w in range(len(cooling)))
        assert active_setpoint
        absent_setpoint = all(all(cooling[w][h] == 30.0
                             for h in range(len(cooling[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(cooling)))
        assert absent_setpoint
