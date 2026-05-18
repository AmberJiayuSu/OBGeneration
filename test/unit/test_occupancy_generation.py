"""
Unit tests for occupancy generator.

Tests the occupancy generation logic in src/generator/occupancy_generator.py
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from obgeneration.generator.occupancy_generator import (
    ClusterAssumptions,
    HouseholdOccupancyFractions,
    OccupancyGenerator,
)
from obgeneration.model.occupancy import (
    Occupancy,
    HouseholdComposition,
    MobilityCluster,
    OccupantMobilityProfile,
    TimeRange,
    WeekdayOccupancyPattern,
    WeekendOccupancyPattern,
    SleepPattern,
    ScheduleRigidness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def single_long_day_away_occupant() -> OccupantMobilityProfile:
    return OccupantMobilityProfile(
        weekday_cluster=MobilityCluster.LONG_DAY_AWAY,
        weekend_cluster=MobilityCluster.MOSTLY_HOME,
    )


@pytest.fixture
def single_long_day_away_occupant() -> OccupantMobilityProfile:
    return OccupantMobilityProfile(
        weekday_cluster=MobilityCluster.LONG_DAY_AWAY,
        weekend_cluster=MobilityCluster.MOSTLY_HOME,
    )


@pytest.fixture
def single_mostly_home_occupant() -> OccupantMobilityProfile:
    return OccupantMobilityProfile(
        weekday_cluster=MobilityCluster.MOSTLY_HOME,
        weekend_cluster=MobilityCluster.MOSTLY_HOME,
    )

@pytest.fixture
def single_night_away_occupant() -> OccupantMobilityProfile:
    return OccupantMobilityProfile(
        weekday_cluster=MobilityCluster.EVENING_NIGHT_AWAY,
        weekend_cluster=MobilityCluster.EVENING_NIGHT_AWAY,
    )


@pytest.fixture
def occupancy_always_home(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household, no away/sleep patterns specified."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(is_always_occupied=True),
        weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True),
        sleep_pattern=SleepPattern(is_always_awake=True)
    )


@pytest.fixture
def occupancy_with_away_cluster_long_day_away(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household with weekday away (8–18) and weekend away (16–20)."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=8, end_hour=18),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        weekend_pattern=WeekendOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=16, end_hour=20),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
    )


@pytest.fixture
def occupancy_with_away_cluster_always_home(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household with weekday away (8–18) and weekend away (16–20)."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=8, end_hour=18),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        weekend_pattern=WeekendOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=16, end_hour=20),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
    )

@pytest.fixture
def occupancy_with_away_night(single_night_away_occupant) -> Occupancy:
    """1-occupant household with weekday away (22–6) that wraps midnight."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_night_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=22, end_hour=6),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        weekend_pattern=WeekendOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour= 23, end_hour=7),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
    )


@pytest.fixture
def occupancy_with_away_and_sleep(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household with sleep pattern (22–06, wraps midnight)."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=8, end_hour=18),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        weekend_pattern=WeekendOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=16, end_hour=20),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        sleep_pattern=SleepPattern(
            is_always_awake=False,
            sleep_time=TimeRange(start_hour=22, end_hour=6),
            sleep_time_rigidness=ScheduleRigidness.MOSTLY_CONSISTENT,
        ),
    )

@pytest.fixture
def occupancy_with_away_and_sleep_overlap(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household with sleep pattern (22–06, wraps midnight)."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=8, end_hour=22),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        weekend_pattern=WeekendOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=16, end_hour=20),
            away_time_rigidness=ScheduleRigidness.SOMEWHAT_VARIABLE,
        ),
        sleep_pattern=SleepPattern(
            is_always_awake=False,
            sleep_time=TimeRange(start_hour=20, end_hour=6),
            sleep_time_rigidness=ScheduleRigidness.MOSTLY_CONSISTENT,
        ),
    )





@pytest.fixture(scope="module")
def default_cluster_assumptions() -> ClusterAssumptions:
    """Load default ClusterAssumptions once per module (disk I/O is slow)."""
    return ClusterAssumptions.default()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


SIM_RES = 15  # minutes — used throughout tests


def _daily_slices_from_annual_schedule(
    annual_schedule: list[float],
    resolution_mins: int,
) -> list[list[float]]:
    bins_per_day = 1440 // resolution_mins
    assert len(annual_schedule) == 365 * bins_per_day
    return [
        annual_schedule[start:start + bins_per_day]
        for start in range(0, len(annual_schedule), bins_per_day)
    ]


def _assert_occupancy_result_structure(result, resolution_mins: int) -> None:
    bins_per_day = 1440 // resolution_mins

    assert len(result.annual_schedule) == 365 * bins_per_day
    assert len(result.occupancy_states) == 53
    assert sum(len(week) for week in result.occupancy_states) == len(result.annual_schedule)
    assert result.winter_design_day_schedule is not None
    assert result.summer_design_day_schedule is not None
    assert len(result.winter_design_day_schedule) == bins_per_day
    assert len(result.summer_design_day_schedule) == bins_per_day


def _assert_design_day_matches_extreme(
    annual_schedule: list[float],
    design_day: list[float] | None,
    mode: str,
    resolution_mins: int,
) -> None:
    daily_schedules = _daily_slices_from_annual_schedule(annual_schedule, resolution_mins)
    assert design_day is not None
    assert any(day == design_day for day in daily_schedules)

    daily_averages = [sum(day) / len(day) for day in daily_schedules]
    design_day_average = sum(design_day) / len(design_day)

    if mode == "max":
        assert design_day_average == pytest.approx(max(daily_averages))
    elif mode == "min":
        assert design_day_average == pytest.approx(min(daily_averages))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

# ---------------------------------------------------------------------------
# OccupancyGenerator — household_away_time_annually
# ---------------------------------------------------------------------------

class TestHouseholdAwayTimeAnnually:
    def test_returns_365_elements(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, rng):
        """Should return exactly 365 elements."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually(rng)
        
        assert isinstance(away_times, list)
        assert len(away_times) == 365

    def test_always_occupied_yields_none(self, occupancy_always_home, default_cluster_assumptions, rng):
        """If is_always_occupied=True, all days should be None."""
        occ = OccupancyGenerator(occupancy_always_home, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually(rng)
        assert len(away_times) == 365
        assert all(at is None for at in away_times)

    def test_away_day (self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, rng):
        """Non-None elements should be TimeRange instances.
            Sampled away intervals should be centered near the specified away_interval mean (statistical)."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually(rng)
        assert len(away_times) == 365
        away_time_weekday = []
        away_time_weekend = []
        for day in range(365):
            at = away_times[day]
            if at is not None:
                assert isinstance(at, TimeRange)
                # Statistical test: sampled times should cluster around the specified means
                if day % 7 in [1, 2, 3, 4, 5]:  # Mon–Fri
                    away_time_weekday.append(at)
                else:
                    away_time_weekend.append(at)

        if away_time_weekday:
            weekday_starts = [at.start_hour for at in away_time_weekday]
            weekday_ends = [at.end_hour for at in away_time_weekday]
            assert np.mean(weekday_starts) == pytest.approx(8.0, abs=1.0)
            assert np.mean(weekday_ends) == pytest.approx(18.0, abs=1.0)
        if away_time_weekend:
            weekend_starts = [at.start_hour for at in away_time_weekend]
            weekend_ends = [at.end_hour for at in away_time_weekend]
            assert np.mean(weekend_starts) == pytest.approx(16.0, abs=1.0)
            assert np.mean(weekend_ends) == pytest.approx(20.0, abs=1.0)

    @pytest.mark.parametrize("rigidness, expected_std", [
        (ScheduleRigidness.STRICT,            0.25),
        (ScheduleRigidness.MOSTLY_CONSISTENT, 0.5),
        (ScheduleRigidness.SOMEWHAT_VARIABLE, 1.0),
        (ScheduleRigidness.HIGHLY_VARIABLE,   2.0),
    ])
    def test_away_time_variance_matches_rigidness(self, default_cluster_assumptions, rigidness, expected_std, rng):
        """Std of sampled start/end hours should approximate the rigidness std_dev
        for both weekday (8–18) and weekend (16–20) away intervals."""
        occupancy = Occupancy(
            num_occupants=1,
            household_composition=HouseholdComposition(occupants=[
                OccupantMobilityProfile(
                    weekday_cluster=MobilityCluster.LONG_DAY_AWAY,
                    weekend_cluster=MobilityCluster.MOSTLY_HOME,
                )
            ]),
            weekday_pattern=WeekdayOccupancyPattern(
                is_always_occupied=False,
                away_interval=TimeRange(start_hour=8, end_hour=18),
                away_time_rigidness=rigidness,
            ),
            weekend_pattern=WeekendOccupancyPattern(
                is_always_occupied=False,
                away_interval=TimeRange(start_hour=16, end_hour=20),
                away_time_rigidness=rigidness,
            ),
        )
        occ = OccupancyGenerator(occupancy, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually(rng)

        wd_starts = [away_times[d].start_hour for d in range(365) if away_times[d] is not None and d % 7 not in [0, 6]]
        wd_ends   = [away_times[d].end_hour   for d in range(365) if away_times[d] is not None and d % 7 not in [0, 6]]
        we_starts = [away_times[d].start_hour for d in range(365) if away_times[d] is not None and d % 7 in  [0, 6]]
        we_ends   = [away_times[d].end_hour   for d in range(365) if away_times[d] is not None and d % 7 in  [0, 6]]

        tol = max(0.15, expected_std * 0.5)

        if wd_starts:
            assert np.std(wd_starts) == pytest.approx(expected_std, abs=tol), (
                f"[{rigidness.value}] weekday start std={np.std(wd_starts):.3f}, expected≈{expected_std}"
            )
            assert np.std(wd_ends) == pytest.approx(expected_std, abs=tol), (
                f"[{rigidness.value}] weekday end std={np.std(wd_ends):.3f}, expected≈{expected_std}"
            )
        if we_starts:
            assert np.std(we_starts) == pytest.approx(expected_std, abs=tol), (
                f"[{rigidness.value}] weekend start std={np.std(we_starts):.3f}, expected≈{expected_std}"
            )
            assert np.std(we_ends) == pytest.approx(expected_std, abs=tol), (
                f"[{rigidness.value}] weekend end std={np.std(we_ends):.3f}, expected≈{expected_std}"
            )

    def test_away_time_wraps_midnight(self, occupancy_with_away_night, default_cluster_assumptions, rng):
        """Test that away_time intervals that wrap midnight are handled correctly."""
        occ = OccupancyGenerator(occupancy_with_away_night, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually(rng)
        assert len(away_times) == 365
        for day in range(365):
            at = away_times[day]
            if at is not None:
                assert isinstance(at, TimeRange)
                if day % 7 in [1, 2, 3, 4, 5]:  # Mon–Fri
                    assert at.wraps_midnight is True
                    assert np.mean([at.start_hour for at in away_times if at is not None and at.wraps_midnight]) == pytest.approx(22.0, abs=1.0)
                    assert np.mean([at.end_hour for at in away_times if at is not None and at.wraps_midnight]) == pytest.approx(6.0, abs=1.0)
                else:
                    assert at.wraps_midnight is True
                    assert np.mean([at.start_hour for at in away_times if at is not None and at.wraps_midnight]) == pytest.approx(23.0, abs=1.0)
                    assert np.mean([at.end_hour for at in away_times if at is not None and at.wraps_midnight]) == pytest.approx(7.0, abs=1.0)

                


# ---------------------------------------------------------------------------
# OccupancyGenerator — household_sleep_time_annually
# ---------------------------------------------------------------------------

class TestHouseholdSleepTimeAnnually:
    def test_returns_with_away_times(self, occupancy_with_away_and_sleep, default_cluster_assumptions, rng):
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        away_time = occ.household_away_time_annually(rng)
        sleep_time = occ.household_sleep_time_annually(rng, away_time)
        assert len(sleep_time) == 365
        start_times = []
        end_times = []
        for day in range(365):
            st = sleep_time[day]
            at = away_time[day]
            if st is not None:
                assert isinstance(st, TimeRange)
                assert st.start_hour >= at.end_hour
                assert st.end_hour <= at.start_hour
                start_times.append(st.start_hour)
                end_times.append(st.end_hour)
        if start_times and end_times:
            assert np.mean(start_times) == pytest.approx(22.0, abs=1.0)
            assert np.mean(end_times) == pytest.approx(6.0, abs=1.0)


    def test_always_awake_yields_all_none(self, occupancy_always_home, default_cluster_assumptions, rng):
        """If is_always_awake=True, all days should be None."""
        occ = OccupancyGenerator(occupancy_always_home, default_cluster_assumptions, SIM_RES)
        away_time = occ.household_away_time_annually(rng)
        sleep_time = occ.household_sleep_time_annually(rng, away_time)
        assert len(sleep_time) == 365
        assert all(st is None for st in sleep_time)

    @pytest.mark.parametrize("rigidness, expected_std", [
        (ScheduleRigidness.STRICT,            0.25),
        (ScheduleRigidness.MOSTLY_CONSISTENT, 0.5),
        (ScheduleRigidness.SOMEWHAT_VARIABLE, 1.0),
        (ScheduleRigidness.HIGHLY_VARIABLE,   2.0),
    ])
    def test_sleep_time_variance_matches_rigidness(self, default_cluster_assumptions, rigidness, expected_std, rng):
        """Std of sampled start/end hours should approximate the rigidness std_dev
        for sleep interval 22–6 (wraps midnight), with no away constraint."""
        occupancy = Occupancy(
            num_occupants=1,
            household_composition=HouseholdComposition(occupants=[
                OccupantMobilityProfile(
                    weekday_cluster=MobilityCluster.LONG_DAY_AWAY,
                    weekend_cluster=MobilityCluster.MOSTLY_HOME,
                )
            ]),
            sleep_pattern=SleepPattern(
                is_always_awake=False,
                sleep_time=TimeRange(start_hour=22, end_hour=6),
                sleep_time_rigidness=rigidness,
            ),
        )
        occ = OccupancyGenerator(occupancy, default_cluster_assumptions, SIM_RES)
        sleep_times = occ.household_sleep_time_annually(rng, [None] * 365)

        starts = [st.start_hour for st in sleep_times if st is not None]
        ends   = [st.end_hour   for st in sleep_times if st is not None]

        assert len(starts) == 365, "Expected all 365 days to have a sleep time (no away constraint)"

        tol = max(0.15, expected_std * 0.5)

        assert np.std(starts) == pytest.approx(expected_std, abs=tol), (
            f"[{rigidness.value}] start std={np.std(starts):.3f}, expected≈{expected_std}"
        )
        assert np.std(ends) == pytest.approx(expected_std, abs=tol), (
            f"[{rigidness.value}] end std={np.std(ends):.3f}, expected≈{expected_std}"
        )

    def test_sleep_no_overlap_with_away(self, occupancy_with_away_and_sleep_overlap, default_cluster_assumptions, rng):
        """Non-None elements should be TimeRange instances."""
        occ = OccupancyGenerator(occupancy_with_away_and_sleep_overlap, default_cluster_assumptions, SIM_RES)
        away_time = occ.household_away_time_annually(rng)
        sleep_time = occ.household_sleep_time_annually(rng, away_time)
        assert len(sleep_time) == 365
        for day in range(365):
            st = sleep_time[day]
            at = away_time[day]
            if st and at:
                assert not st.overlaps(at)


# ---------------------------------------------------------------------------
# OccupancyGenerator — household_mc_state_annually
# ---------------------------------------------------------------------------
class TestHouseholdMCStateAnnually:
    def test_returns(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, rng):
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        states = occ.household_mc_state_annually(rng)
        assert isinstance(states, list)
        assert len(states) == 53
        for week in range(53):
            assert isinstance(states[week], list)
            if week < 52:
                assert len(states[week]) == 7 * (1440 // SIM_RES)
            else:
                assert len(states[week]) == 1 * (1440 // SIM_RES)
            for s in states[week]:
                assert isinstance(s, HouseholdOccupancyFractions)



# ---------------------------------------------------------------------------
# OccupancyGenerator — apply_away_time
# ---------------------------------------------------------------------------

class TestApplyAwayTime:
    def _make_all_home_states(self, num_weeks=53, bins_per_week=7 * 96) -> list[list[HouseholdOccupancyFractions]]:
        """Helper: all bins set to fully home."""
        return [[HouseholdOccupancyFractions(home=1.0, sleep=0.0)] * bins_per_week for _ in range(num_weeks)]

    def _make_all_away_states(self, num_weeks=53, bins_per_week=7 * 96) -> list[list[HouseholdOccupancyFractions]]:
        """Helper: all bins set to fully away."""
        return [[HouseholdOccupancyFractions(home=0.0, sleep=0.0)] * bins_per_week for _ in range(num_weeks)]

    def test_away_bins_forced_to_zero(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, rng):
        """Bins inside the away_time window must have home=0, sleep=0."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0

        # Fixed (no-variance) away intervals for deterministic checking
        wd_away = TimeRange(start_hour=8, end_hour=18)
        we_away = TimeRange(start_hour=16, end_hour=20)
        away_time = [we_away if (d % 7) in [0, 6] else wd_away for d in range(365)]

        states = occ.apply_away_time(occ.household_mc_state_annually(rng), away_time)

        for day in range(365):
            away_range = away_time[day]
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            for b in range(bins_per_day):
                f = states[week][day_start + b]
                if away_range.contains_hour(b * resolution_hours):
                    assert f.home + f.sleep == 0.0, (
                        f"Day {day} bin {b} ({b * resolution_hours:.2f}h): expected home+sleep=0, got {f.home + f.sleep}"
                    )


    def test_outside_away_bins_unchanged_if_nonzero(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
        """Bins outside the away window with home>0 should not be modified."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0

        wd_away = TimeRange(start_hour=8, end_hour=18)
        we_away = TimeRange(start_hour=16, end_hour=20)
        away_time = [we_away if (d % 7) in [0, 6] else wd_away for d in range(365)]

        initial_states = self._make_all_home_states()  # home=1.0 everywhere
        result = occ.apply_away_time(initial_states, away_time)

        for day in range(365):
            away_range = away_time[day]
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            for b in range(bins_per_day):
                f = result[week][day_start + b]
                if not away_range.contains_hour(b * resolution_hours):
                    assert f.home == 1.0 and f.sleep == 0.0, (
                        f"Day {day} bin {b}: outside-away bin was modified (home={f.home}, sleep={f.sleep})"
                    )

    def test_outside_away_bins_get_min_home_when_both_zero(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
        """Bins outside the away window that are fully away get raised to one_unit home."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0
        one_unit = 1.0 / occupancy_with_away_cluster_long_day_away.num_occupants

        wd_away = TimeRange(start_hour=8, end_hour=18)
        we_away = TimeRange(start_hour=16, end_hour=20)
        away_time = [we_away if (d % 7) in [0, 6] else wd_away for d in range(365)]

        initial_states = self._make_all_away_states()  # home=0.0, sleep=0.0 everywhere
        result = occ.apply_away_time(initial_states, away_time)

        for day in range(365):
            away_range = away_time[day]
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            for b in range(bins_per_day):
                f = result[week][day_start + b]
                if not away_range.contains_hour(b * resolution_hours):
                    assert f.home == one_unit and f.sleep == 0.0, (
                        f"Day {day} bin {b}: expected home={one_unit}, got home={f.home}, sleep={f.sleep}"
                    )

    def test_none_away_day_leaves_states_unchanged(self, occupancy_always_home, default_cluster_assumptions):
        """When away_time for a day is None, no states are modified."""
        occ = OccupancyGenerator(occupancy_always_home, default_cluster_assumptions, SIM_RES)
        initial_states = self._make_all_home_states()
        expected = [[f for f in week] for week in initial_states]

        result = occ.apply_away_time(initial_states, [None] * 365)

        for w, week in enumerate(result):
            for b, f in enumerate(week):
                e = expected[w][b]
                assert f.home == e.home and f.sleep == e.sleep, (
                    f"Week {w} bin {b}: state changed despite None away_time"
                )

    def test_average_away_bins_matches_interval(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, rng):
        """Average number of away bins per day (counted from processed states) should approximate
        the expected bin count for the specified away interval, for weekday and weekend separately.

        Weekday 8–18  → 10 hrs × 4 bins/hr = 40 bins
        Weekend 16–20 →  4 hrs × 4 bins/hr = 16 bins

        After apply_away_time, bins outside the away window are guaranteed home+sleep > 0,
        so counting home+sleep == 0 per day directly measures the applied away window length.
        With SOMEWHAT_VARIABLE (std_dev=1 hr), the SEM across ~260 weekdays / ~104 weekends
        is < 0.5 bins, so a tolerance of ±3 bins is well within 6-sigma.
        """
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES

        away_time = occ.household_away_time_annually(rng)
        states = occ.apply_away_time(occ.household_mc_state_annually(rng), away_time)

        wd_away_counts = []
        we_away_counts = []
        for day in range(365):
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            n_away = sum(
                1 for b in range(bins_per_day)
                if states[week][day_start + b].home + states[week][day_start + b].sleep == 0.0
            )
            if day % 7 in [0, 6]:
                we_away_counts.append(n_away)
            else:
                wd_away_counts.append(n_away)

        expected_wd = int((18 - 8) * 60 / SIM_RES)   # 40 bins
        expected_we = int((20 - 16) * 60 / SIM_RES)  # 16 bins

        assert np.mean(wd_away_counts) == pytest.approx(expected_wd, abs=3), (
            f"Weekday avg away bins: {np.mean(wd_away_counts):.1f}, expected ~{expected_wd}"
        )
        assert np.mean(we_away_counts) == pytest.approx(expected_we, abs=3), (
            f"Weekend avg away bins: {np.mean(we_away_counts):.1f}, expected ~{expected_we}"
        )


# ---------------------------------------------------------------------------
# OccupancyGenerator — apply_sleep_time
# ---------------------------------------------------------------------------

class TestApplySleepTime:

    def _make_all_home_states(self, num_weeks=53, bins_per_week=7 * 96) -> list[list[HouseholdOccupancyFractions]]:
        """Helper: all bins set to fully home (home=1.0, sleep=0.0)."""
        return [[HouseholdOccupancyFractions(home=1.0, sleep=0.0)] * bins_per_week for _ in range(num_weeks)]

    def _make_states_with_sleep(self, home: float, sleep: float, num_weeks=53, bins_per_week=7 * 96) -> list[list[HouseholdOccupancyFractions]]:
        """Helper: all bins set to the given home/sleep fractions."""
        return [[HouseholdOccupancyFractions(home=home, sleep=sleep)] * bins_per_week for _ in range(num_weeks)]

    def test_sleep_bins_forced_to_all_sleep(self, occupancy_with_away_and_sleep, default_cluster_assumptions, rng):
        """Bins inside sleep_time must have home=0, sleep=1.0."""
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0

        sleep_range = TimeRange(start_hour=22, end_hour=6)  # fixed, no variance
        sleep_time = [sleep_range] * 365

        states = occ.apply_sleep_time(occ.household_mc_state_annually(rng), sleep_time)

        for day in range(365):
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            for b in range(bins_per_day):
                f = states[week][day_start + b]
                if sleep_range.contains_hour(b * resolution_hours):
                    assert f.home == 0.0 and f.sleep == 1.0, (
                        f"Day {day} bin {b} ({b * resolution_hours:.2f}h): expected home=0, sleep=1, got {f}"
                    )

    def test_outside_sleep_bins_clamp_excess_sleep(self, occupancy_with_away_and_sleep, default_cluster_assumptions):
        """Bins outside sleep_time with sleep > max_awake_sleep_ratio should be clamped to 0,
        and the excess redistributed to home."""
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0

        sleep_range = TimeRange(start_hour=22, end_hour=6)
        sleep_time = [sleep_range] * 365

        # Start with states that have sleep=0.5 everywhere (excess sleep outside window)
        initial_home, initial_sleep = 0.3, 0.5
        initial_states = self._make_states_with_sleep(home=initial_home, sleep=initial_sleep)
        result = occ.apply_sleep_time(initial_states, sleep_time)

        for day in range(365):
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            for b in range(bins_per_day):
                f = result[week][day_start + b]
                if not sleep_range.contains_hour(b * resolution_hours):
                    if f.sleep + f.home > 0.0:  # Only check bins that had some occupancy to start with
                        ratio = f.sleep / (f.sleep + f.home)
                        assert ratio <= 0.3, (
                            f"Day {day} bin {b}: sleep not clamped (sleep={f.sleep})"
                        )
                        assert (f.home + f.sleep) == pytest.approx(initial_home + initial_sleep), (
                            f"Day {day} bin {b}: excess sleep not redistributed to home (home={f.home})"
                        )
                else:
                    assert f.home == 0.0 and f.sleep == 1.0, (
                        f"Day {day} bin {b}: expected home=0, sleep=1 inside sleep window, got {f}"
                    )

    def test_none_sleep_day_leaves_states_unchanged(self, occupancy_with_away_and_sleep, default_cluster_assumptions):
        """When sleep_time for a day is None, no states are modified."""
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        initial_states = self._make_all_home_states()
        expected = [[f for f in week] for week in initial_states]

        result = occ.apply_sleep_time(initial_states, [None] * 365)

        for w, week in enumerate(result):
            for b, f in enumerate(week):
                e = expected[w][b]
                assert f.home == e.home and f.sleep == e.sleep, (
                    f"Week {w} bin {b}: state changed despite None sleep_time"
                )

    def test_average_sleep_bins_matches_interval(self, occupancy_with_away_and_sleep, default_cluster_assumptions, rng):
        """Average number of sleep bins per day (sleep==1.0 after apply_sleep_time) should
        approximate the expected bin count for the specified sleep interval.

        Sleep 22–6 (wraps midnight) → 8 hrs × 4 bins/hr = 32 bins/day.

        After apply_sleep_time with max_awake_sleep_ratio=0.0, sleep==1.0 marks exactly the
        sleep window and sleep==0.0 everywhere else, so the count is a clean proxy.
        With MOSTLY_CONSISTENT (std_dev=0.5 hr), SEM across 365 days is ~0.4 bins,
        so a tolerance of ±3 bins is well within 7-sigma.
        """
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES

        sleep_time = occ.household_sleep_time_annually(rng, [None] * 365)
        states = occ.apply_sleep_time(occ.household_mc_state_annually(rng), sleep_time)

        daily_sleep_counts = []
        for day in range(365):
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            n_sleep = sum(
                1 for b in range(bins_per_day)
                if states[week][day_start + b].sleep == 1.0
            )
            daily_sleep_counts.append(n_sleep)

        # Sleep 22–6: contains_hour uses [start, end), wraps midnight
        # 22*4=88 to 24*4=96 → 8 bins; 0*4=0 to 6*4=24 → 24 bins; total 32 bins
        expected_sleep_bins = int((24 - 22 + 6) * 60 / SIM_RES)  # 32 bins

        assert np.mean(daily_sleep_counts) == pytest.approx(expected_sleep_bins, abs=3), (
            f"Avg sleep bins: {np.mean(daily_sleep_counts):.1f}, expected ~{expected_sleep_bins}"
        )



# ---------------------------------------------------------------------------
# OccupancyGenerator — generate (integration)
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_3_people(self, default_cluster_assumptions, rng):
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
        occ = Occupancy.model_validate_json(occ_json)
        occ_gen = OccupancyGenerator(occ, default_cluster_assumptions, SIM_RES)
        schedule = occ_gen.occupancy_annual_schedule(rng)
        occ_schedule = OccupancyGenerator.to_occupancy_schedule(schedule)

        matplotlib.use("Agg")
        output_dir = Path(__file__).parents[0] / "output"
        output_dir.mkdir(exist_ok=True)

        bins_per_day = 1440 // SIM_RES  # 96
        grid = np.array(occ_schedule[:365 * bins_per_day]).reshape(365, bins_per_day).T  # (96, 365)

        y_ticks = list(range(0, bins_per_day, bins_per_day // 8))
        y_labels = [f"{int(t * SIM_RES / 60):02d}:00" for t in y_ticks]

        fig, ax = plt.subplots(figsize=(18, 6))
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0,
                       origin="upper", extent=[0, 365, bins_per_day, 0])
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Time of Day")
        ax.set_title("Occupancy Schedule — Fraction at Home or Asleep")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        fig.colorbar(im, ax=ax, label="Occupied Fraction (home + sleep)")

        out_path = output_dir / "occ_schedule_heatmap_3ppl.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        assert out_path.exists()


class TestGenerateResult:
    def test_design_days_and_structure(self, default_cluster_assumptions, rng):
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
        occupancy = Occupancy.model_validate_json(occ_json)

        result = OccupancyGenerator.generate_result(
            occupancy=occupancy,
            cluster_assumptions=default_cluster_assumptions,
            resolution_mins=SIM_RES,
            rng=rng,
        )

        _assert_occupancy_result_structure(result, SIM_RES)
        assert result.annual_schedule == OccupancyGenerator.to_occupancy_schedule(result.occupancy_states)
        _assert_design_day_matches_extreme(
            result.annual_schedule,
            result.summer_design_day_schedule,
            mode="max",
            resolution_mins=SIM_RES,
        )
        _assert_design_day_matches_extreme(
            result.annual_schedule,
            result.winter_design_day_schedule,
            mode="min",
            resolution_mins=SIM_RES,
        )





# ---------------------------------------------------------------------------
# OccupancyGenerator — static methods
# ---------------------------------------------------------------------------

class TestActiveSleepMask:

    def _states(self, home: float, sleep: float, weeks: int = 2, bins: int = 4) -> list[list[HouseholdOccupancyFractions]]:
        return [[HouseholdOccupancyFractions(home=home, sleep=sleep)] * bins for _ in range(weeks)]

    def test_all_away_produces_all_false_masks(self):
        """Fully-away bins (home=0, sleep=0) → both masks all False."""
        states = self._states(home=0.0, sleep=0.0)
        active, sleep = OccupancyGenerator.active_sleep_mask(states)
        assert all(v is False for week in active for v in week)
        assert all(v is False for week in sleep  for v in week)

    def test_high_home_ratio_is_active(self):
        """home=0.8, sleep=0.1 → ratio≈0.89 ≥ default threshold 0.3 → active."""
        states = self._states(home=0.8, sleep=0.1)
        active, sleep = OccupancyGenerator.active_sleep_mask(states)
        assert all(v is True  for week in active for v in week)
        assert all(v is False for week in sleep  for v in week)

    def test_low_home_ratio_is_sleep(self):
        """home=0.05, sleep=0.9 → ratio≈0.05 < 0.3 → sleep."""
        states = self._states(home=0.05, sleep=0.9)
        active, sleep = OccupancyGenerator.active_sleep_mask(states)
        assert all(v is False for week in active for v in week)
        assert all(v is True  for week in sleep  for v in week)

    def test_threshold_boundary_is_active(self):
        """ratio exactly at threshold (home=0.3, sleep=0.7 → ratio=0.3) → active."""
        states = self._states(home=0.3, sleep=0.7)
        active, sleep = OccupancyGenerator.active_sleep_mask(states)
        assert all(v is True  for week in active for v in week)
        assert all(v is False for week in sleep  for v in week)

    def test_just_below_threshold_is_sleep(self):
        """ratio just below threshold (home=0.29, sleep=0.71 → ratio<0.3) → sleep."""
        states = self._states(home=0.29, sleep=0.71)
        active, sleep = OccupancyGenerator.active_sleep_mask(states)
        assert all(v is False for week in active for v in week)
        assert all(v is True  for week in sleep  for v in week)

    def test_masks_are_mutually_exclusive(self):
        """No bin can be both active and sleep simultaneously."""
        mixed = [
            [
                HouseholdOccupancyFractions(home=0.8, sleep=0.1),
                HouseholdOccupancyFractions(home=0.0, sleep=0.9),
                HouseholdOccupancyFractions(home=0.0, sleep=0.0),
            ]
        ]
        active, sleep = OccupancyGenerator.active_sleep_mask(mixed)
        for w in range(len(active)):
            for t in range(len(active[w])):
                assert not (active[w][t] and sleep[w][t]), f"Week {w} bin {t} is both active and sleep"

    def test_output_shape_matches_input(self):
        """active_mask and sleep_mask must have the same shape as occ_states."""
        states = self._states(home=0.5, sleep=0.3, weeks=3, bins=7)
        active, sleep = OccupancyGenerator.active_sleep_mask(states)
        assert len(active) == len(states)
        assert len(sleep)  == len(states)
        for w in range(len(states)):
            assert len(active[w]) == len(states[w])
            assert len(sleep[w])  == len(states[w])

    def test_custom_threshold(self):
        """home=0.4, sleep=0.6 → ratio≈0.4; below threshold=0.5 → sleep, above threshold=0.3 → active."""
        states = self._states(home=0.4, sleep=0.6)
        active_low,  sleep_low  = OccupancyGenerator.active_sleep_mask(states, active_threshold=0.3)
        active_high, sleep_high = OccupancyGenerator.active_sleep_mask(states, active_threshold=0.5)
        assert all(v is True  for week in active_low  for v in week)
        assert all(v is False for week in sleep_low   for v in week)
        assert all(v is False for week in active_high for v in week)
        assert all(v is True  for week in sleep_high  for v in week)


class TestApplyToMask:

    def test_masked_bins_get_value(self):
        """Bins where mask=True must receive the specified value."""
        mask     = [True, False, True, False]
        schedule = [0.1,  0.5,   0.9,  0.3]
        result   = OccupancyGenerator.apply_to_mask(mask, schedule, value=99.0)
        assert result[0] == pytest.approx(99.0)
        assert result[2] == pytest.approx(99.0)

    def test_unmasked_bins_unchanged(self):
        """Bins where mask=False must retain their original value."""
        mask     = [True, False, True, False]
        schedule = [0.1,  0.5,   0.9,  0.3]
        result   = OccupancyGenerator.apply_to_mask(mask, schedule, value=99.0)
        assert result[1] == pytest.approx(0.5)
        assert result[3] == pytest.approx(0.3)

    def test_mismatched_lengths_raise(self):
        """Passing mask and schedule with different lengths must raise AssertionError."""
        with pytest.raises(AssertionError):
            OccupancyGenerator.apply_to_mask([True, False], [0.1, 0.2, 0.3], value=1.0)

    def test_all_false_mask_returns_original(self):
        """All-False mask → result equals the original schedule."""
        schedule = [0.1, 0.4, 0.7, 1.0]
        result   = OccupancyGenerator.apply_to_mask([False] * 4, schedule, value=99.0)
        assert result == pytest.approx(schedule)

    def test_all_true_mask_returns_all_value(self):
        """All-True mask → every element equals value."""
        schedule = [0.1, 0.4, 0.7, 1.0]
        result   = OccupancyGenerator.apply_to_mask([True] * 4, schedule, value=0.0)
        assert all(v == pytest.approx(0.0) for v in result)

    def test_output_length_matches_input(self):
        """Result length must equal the input schedule length."""
        schedule = [float(i) for i in range(10)]
        mask     = [i % 2 == 0 for i in range(10)]
        result   = OccupancyGenerator.apply_to_mask(mask, schedule, value=-1.0)
        assert len(result) == len(schedule)
