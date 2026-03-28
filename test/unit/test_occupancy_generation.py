"""
Unit tests for occupancy generator.

Tests the occupancy generation logic in src/generator/occupancy_generator.py
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from ob_generation.generator.occupancy_generator import (
    ClusterAssumptions,
    HouseholdOccupancyFractions,
    TimeRangeDistribution,
    OccupancyGenerator,
)
from ob_generation.model.occupancy import (
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
def away_range_typical() -> TimeRange:
    """Typical away range (9–17) that does not wrap midnight."""
    return TimeRange(start_hour=9, end_hour=17)

@pytest.fixture
def away_range_wraps_midnight() -> TimeRange:
    """Away range (22–6) that wraps midnight."""
    return TimeRange(start_hour=22, end_hour=6)

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
        weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True)
    )


@pytest.fixture
def occupancy_with_away(single_long_day_away_occupant) -> Occupancy:
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
def occupancy_with_sleep(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household with sleep pattern (22–06, wraps midnight)."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        sleep_pattern=SleepPattern(
            is_always_awake=False,
            sleep_time=TimeRange(start_hour=22, end_hour=6),
            sleep_time_rigidness=ScheduleRigidness.MOSTLY_CONSISTENT,
        ),
    )


@pytest.fixture
def occupancy_with_away_and_sleep(single_long_day_away_occupant) -> Occupancy:
    """1-occupant household with both away and sleep patterns."""
    return Occupancy(
        num_occupants=1,
        household_composition=HouseholdComposition(occupants=[single_long_day_away_occupant]),
        weekday_pattern=WeekdayOccupancyPattern(
            is_always_occupied=False,
            away_interval=TimeRange(start_hour=8, end_hour=18),
        ),
        weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True),
        sleep_pattern=SleepPattern(
            is_always_awake=False,
            sleep_time=TimeRange(start_hour=22, end_hour=6),
        ),
    )


@pytest.fixture
def occupancy_two_occupants() -> Occupancy:
    """2-occupant household with different mobility clusters."""
    return Occupancy(
        num_occupants=2,
        household_composition=HouseholdComposition(occupants=[
            OccupantMobilityProfile(
                weekday_cluster=MobilityCluster.LONG_DAY_AWAY,
                weekend_cluster=MobilityCluster.MOSTLY_HOME,
            ),
            OccupantMobilityProfile(
                weekday_cluster=MobilityCluster.MOSTLY_HOME,
                weekend_cluster=MobilityCluster.AFTERNOON_AWAY,
            ),
        ]),
    )


@pytest.fixture(scope="module")
def default_cluster_assumptions() -> ClusterAssumptions:
    """Load default ClusterAssumptions once per module (disk I/O is slow)."""
    return ClusterAssumptions.default()


SIM_RES = 15  # minutes — used throughout tests


# ---------------------------------------------------------------------------
# TimeRangeDistribution
# ---------------------------------------------------------------------------



class TestTimeRange:
    @pytest.mark.parametrize("value, resolution, expected", [
        (8.1, 0.25, 8.0),
        (8.2, 0.25, 8.25),
        (8.4, 0.25, 8.5),
        (0.1, 0.5, 0.0),
        (23.9, 1.0, 23.0),
        (12.33, 0.1, 12.3),
        (5.67, 0.2, 5.6)
    ])
    def test_snap_to_resolution(self, value, resolution, expected):
        """Test that time snapping works correctly."""
        assert TimeRangeDistribution._snap_to_resolution(value, resolution) == pytest.approx(expected)

    @pytest.mark.parametrize("start, end", [
        (22.0, 2.0),
        (23.5, 0.5),
    ])
    def test_wraps_midnight_handling(self, start, end):
        """Test that wraps_midnight flag is respected."""
        wrapping_range = TimeRange(start_hour=start, end_hour=end)
        assert wrapping_range.wraps_midnight is True

        dist = TimeRangeDistribution(wrapping_range, start_variance=0.5, end_variance=0.5, resolution_mins=15)
        assert dist.wraps_midnight is True
        sampled = dist.sample()
        assert isinstance(sampled, TimeRange)


    @pytest.mark.statistical
    @pytest.mark.parametrize("start, end", [
        (9.0, 17.0),
        (8.5, 16.5),
        (10.0, 15.0)
    ])    
    def test_sample_returns_valid_time_range(self, start, end):
        """Test that sample returns a valid TimeRange."""
        base_range = TimeRange(start_hour=start, end_hour=end)
        dist = TimeRangeDistribution(base_range, start_variance=1.0, end_variance=1.0, resolution_mins=15)

        starts = []
        ends = []
        for _ in range(100):
            sampled = dist.sample()
            starts.append(sampled.start_hour)
            ends.append(sampled.end_hour)
            assert isinstance(sampled, TimeRange)
            assert 0 <= sampled.start_hour < 24
            assert 0 <= sampled.end_hour < 24
            # Non-wrapping ranges should have end > start
            if not sampled.wraps_midnight:
                assert sampled.end_hour > sampled.start_hour

        start_mean = np.mean(starts)
        end_mean = np.mean(ends)
        assert abs(start_mean - start) / abs(start) < 0.05
        assert abs(end_mean - end) / abs(end) < 0.05

        start_std = np.std(starts)
        end_std = np.std(ends)
        assert abs(start_std - 1.0) < 0.3
        assert abs(end_std - 1.0) < 0.3


    @pytest.mark.parametrize("start, end, res, var", [
        (22.0,4.0,15,1.0),
        (9.15,17.45,30,3.0),
        (8.5,16.5,60,2.0)
    ])
    def test_sample_many_times_stays_valid(self, start, end, res, var):
        """Run 100 samples and verify each satisfies all constraints."""
        base_range = TimeRange(start_hour=start, end_hour=end)
        dist = TimeRangeDistribution(base_range, start_variance=var, end_variance=var, resolution_mins=res)
        start_vals = []
        end_vals = []
        for _ in range(100):
            sampled = dist.sample()
            start_vals.append(sampled.start_hour)
            end_vals.append(sampled.end_hour)
            assert isinstance(sampled, TimeRange)
            assert 0 <= sampled.start_hour < 24
            assert 0 <= sampled.end_hour < 24
            if not sampled.wraps_midnight:
                assert sampled.end_hour > sampled.start_hour
        assert np.mean(start_vals) == pytest.approx(start, abs=0.5)
        assert np.mean(end_vals) == pytest.approx(end, abs=0.5)
        assert np.std(start_vals) == pytest.approx(var, abs=0.5)
        assert np.std(end_vals) == pytest.approx(var, abs=0.5)

        

class TestTimeRangeDistributionSampleWithAwayBounds:
    @pytest.mark.parametrize("start, end", [
        (10.0, 18.0),
        (7.0, 14.0)
    ])
    def test_returns_none_when_mean_in_away_range_day(self, start, end, away_range_typical):
        """Returns None when the distribution mean falls inside the away window."""
        away_range = away_range_typical
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(away_range)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (23.0, 8.0),
        (21.0,4.0)
    ])
    def test_returns_none_when_mean_in_away_range_night(self, start, end, away_range_wraps_midnight):
        """Returns None when the distribution mean falls inside the away window."""
        away_range = away_range_wraps_midnight
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(away_range)
        print(f"Sampled range: {sampled}, away range: {away_range}")
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (11.0, 14.0),
        (7.0, 20.0)
    ])
    def test_returns_none_when_fully_contains_day(self, start, end, away_range_typical):
        """Returns None when away_time fully contains the TimeRange."""
        away_range = away_range_typical
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(away_range)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (19.0, 8.0),
        (1.0, 5.0)
    ])
    def test_returns_none_when_fully_contains_night(self, start, end, away_range_wraps_midnight):
        """Returns None when away_time fully contains the TimeRange."""
        away_range = away_range_wraps_midnight
        dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
        sampled = dist.sample_with_away_bounds(away_range)
        assert sampled is None

    @pytest.mark.parametrize("start, end", [
        (22.0, 6.0),
        (1.0, 8.0)
    ])
    def test_sampled_range_outside_away_window_day(self,start,end,away_range_typical):
        """Returned TimeRange should not overlap with the away_time window."""
        for _ in range(50):
            away_range = away_range_typical
            dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
            sampled = dist.sample_with_away_bounds(away_range)
            assert sampled is not None
            assert not sampled.overlaps(away_range)

    @pytest.mark.parametrize("start, end", [
        (8.0, 15.0),
        (12.0, 20.0)
    ])
    def test_sampled_range_outside_away_window_night(self,start,end,away_range_wraps_midnight):
        """Returned TimeRange should not overlap with the away_time window."""
        for _ in range(50):
            away_range = away_range_wraps_midnight
            dist = TimeRangeDistribution(TimeRange(start_hour=start, end_hour=end), start_variance=1.0, end_variance=1.0, resolution_mins=SIM_RES)
            sampled = dist.sample_with_away_bounds(away_range)
            assert sampled is not None
            assert not sampled.overlaps(away_range)




# ---------------------------------------------------------------------------
# ClusterAssumptions
# ---------------------------------------------------------------------------

class TestClusterAssumptions:
    def test_default_loads(self, default_cluster_assumptions):
        """ClusterAssumptions.default() should load all CSV data successfully.
            5 clusters,
            weekday_initial_probs: list[list[float]], shape [5, 3],
            weekday_transition_probs: dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]},
            weekend_initial_probs: list[list[float]], shape [5, 3],
            weekend_transition_probs: dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]}.
            Each row of probabilities should sum to ~1.0."""
        ca = default_cluster_assumptions
        assert ca.num_clusters == 5

        # initial probs: list[list[float]], shape [5, 3]
        assert len(ca.weekday_initial_probs) == 5
        assert all(len(row) == 3 for row in ca.weekday_initial_probs)
        assert len(ca.weekend_initial_probs) == 5
        assert all(len(row) == 3 for row in ca.weekend_initial_probs)

        # transition probs: dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]}
        assert len(ca.weekday_transition_probs) == 5
        assert len(ca.weekend_transition_probs) == 5

        for i in range(5):
            assert sum(ca.weekday_initial_probs[i]) == pytest.approx(1.0)
            assert sum(ca.weekend_initial_probs[i]) == pytest.approx(1.0)
            for bin_matrix in ca.weekday_transition_probs[i]:
                for row in bin_matrix:
                    assert sum(row) == pytest.approx(1.0)
            for bin_matrix in ca.weekend_transition_probs[i]:
                for row in bin_matrix:
                    assert sum(row) == pytest.approx(1.0)

    def test_same_resolution_returns_original_shape(self, default_cluster_assumptions):
        """With resolution == assumption_resolution_min, shape should be unchanged."""
        ca = default_cluster_assumptions
        num_bins = 1440 // ca.assumption_resolution_min  #
        matrix = np.array(ca.weekday_transition_probs[0])  # (96, 3, 3)
        result = ca._build_cumsum(matrix, ca.assumption_resolution_min)
        assert result.shape == (num_bins, 3, 3)

    def test_upsample_doubles_bins(self):
        """Halving resolution_min should double the number of bins."""
        assumption_res = 30
        num_bins = 1440 // assumption_res  # 48
        uniform_row = [1/3, 1/3, 1/3]
        trans_probs = {i: [[uniform_row, uniform_row, uniform_row]] * num_bins for i in range(5)}
        init_probs = [[1/3, 1/3, 1/3]] * 5
        ca = ClusterAssumptions(5, 240, assumption_res, init_probs, init_probs, trans_probs, trans_probs)
        matrix = np.array(trans_probs[0])  # (48, 3, 3)
        result = ca._build_cumsum(matrix, assumption_res // 2)  # resolution_min=15
        assert result.shape[0] == num_bins * 2  # 96

    def test_downsample_halves_bins(self, default_cluster_assumptions):
        """Doubling resolution_min should halve the number of bins."""
        ca = default_cluster_assumptions
        num_bins = 1440 // ca.assumption_resolution_min  # 96
        matrix = np.array(ca.weekday_transition_probs[0])  # (96, 3, 3)
        result = ca._build_cumsum(matrix, ca.assumption_resolution_min * 2)  # resolution_min=30
        assert result.shape[0] == num_bins // 2  # 48

    def test_cumsum_last_value_is_one(self, default_cluster_assumptions):
        """Last value along the last axis (cumsum over to-states) should be ~1.0."""
        ca = default_cluster_assumptions
        matrix = np.array(ca.weekday_transition_probs[0])  # (96, 3, 3), rows sum to 1
        result = ca._build_cumsum(matrix, ca.assumption_resolution_min)
        assert result[:, :, -1] == pytest.approx(np.ones((result.shape[0], 3)))





class TestSampleCluster:
    def test_returns_53_weeks(self, default_cluster_assumptions):
        """sample_cluster_annually should return a list of 53 weeks."""
        weeks = default_cluster_assumptions.sample_cluster_annually(0,0,SIM_RES)
        assert isinstance(weeks, list)
        assert len(weeks) == 53
        for w in range(53):
            week = weeks[w]
            assert isinstance(week, list)
            if w < 52:
                assert len(week) == 7 * (1440 // SIM_RES)
            else:
                assert len(week) == 1 * (1440 // SIM_RES)
                
    def test_output_plot(self, default_cluster_assumptions):
        """Plot the sampled cluster as a heatmap (days x time-of-day) and save to output/."""
        matplotlib.use("Agg")

        output_dir = Path(__file__).parents[0] / "output"
        output_dir.mkdir(exist_ok=True)

        bins_per_day = 1440 // SIM_RES  # 96 bins at 15-min resolution
        state_labels = {0: "Away", 1: "Home", 2: "Sleep"}
        cmap = matplotlib.colors.ListedColormap(["#d62728", "#2ca02c", "#1f77b4"])  # Away=red, Home=green, Sleep=blue
        norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        y_ticks = list(range(0, bins_per_day, bins_per_day // 8))
        y_labels = [f"{int(t * SIM_RES / 60):02d}:00" for t in y_ticks]

        cluster_names = ["mostly_home", "long_day_away", "morning_away", "afternoon_away", "evening_night_away"]

        for cluster in range(5):
            weeks = default_cluster_assumptions.sample_cluster_annually(cluster, cluster, SIM_RES)

            # Flatten 53-week structure into (365, bins_per_day), then transpose to (bins_per_day, 365)
            flat = [s.value for week in weeks for s in week][:365 * bins_per_day]
            grid = np.array(flat, dtype=np.int8).reshape(365, bins_per_day).T  # shape (96, 365)

            fig, ax = plt.subplots(figsize=(18, 6))
            im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, origin="upper",
                           extent=[0, 365, bins_per_day, 0])

            ax.set_xlabel("Day of Year")
            ax.set_ylabel("Time of Day")
            ax.set_title(f"Occupancy States — Cluster {cluster + 1} ({cluster_names[cluster]})")
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)

            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels([state_labels[i] for i in range(3)])

            out_path = output_dir / f"cluster_{cluster + 1:02d}_heatmap.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            assert out_path.exists()

    def test_average_occupancy_line_plot(self, default_cluster_assumptions):
        """Plot average (home + sleep) fraction by time of day for weekdays and weekends, one line per cluster."""
        matplotlib.use("Agg")

        output_dir = Path(__file__).parents[0] / "output"
        output_dir.mkdir(exist_ok=True)

        bins_per_day = 1440 // SIM_RES  # 96 bins
        cluster_names = ["mostly_home", "long_day_away", "morning_away", "afternoon_away", "evening_night_away"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # x-axis tick positions and labels (every 3 hours)
        x_ticks = list(range(0, bins_per_day, bins_per_day // 8))
        x_labels = [f"{int(t * SIM_RES / 60):02d}:00" for t in x_ticks]

        # day 0 = Sunday; weekends are day%7 in {0,6}
        day_indices = np.arange(365)
        weekday_mask = ~np.isin(day_indices % 7, [0, 6])  # Mon–Fri
        weekend_mask = np.isin(day_indices % 7, [0, 6])   # Sat–Sun

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        fig.suptitle("Average Occupied Fraction by Time of Day (Home + Sleep)")

        for cluster in range(5):
            weeks = default_cluster_assumptions.sample_cluster_annually(cluster, cluster, SIM_RES)

            flat = np.array([s.value for week in weeks for s in week][:365 * bins_per_day], dtype=np.int8)
            grid = flat.reshape(365, bins_per_day)  # shape (365, bins_per_day)

            # occupied = HOME (1) or SLEEP (2); AWAY (0) = not occupied
            occupied = (grid >= 1).astype(float)  # shape (365, bins_per_day)

            wd_avg = occupied[weekday_mask].mean(axis=0)  # shape (bins_per_day,)
            we_avg = occupied[weekend_mask].mean(axis=0)

            label = f"C{cluster + 1}: {cluster_names[cluster]}"
            axes[0].plot(wd_avg, color=colors[cluster], label=label)
            axes[1].plot(we_avg, color=colors[cluster], label=label)

        for ax, title in zip(axes, ["Weekday", "Weekend"]):
            ax.set_title(title)
            ax.set_xlabel("Time of Day")
            ax.set_ylabel("Avg Occupied Fraction")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        out_path = output_dir / "cluster_avg_occupancy_line.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        assert out_path.exists()





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

    def test_away_bins_forced_to_zero(self, occupancy_with_away, default_cluster_assumptions):
        """Bins inside the away_time window must have home=0, sleep=0."""
        pass

    def test_outside_away_bins_unchanged_if_nonzero(self, occupancy_with_away, default_cluster_assumptions):
        """Bins outside the away window with home>0 should not be modified."""
        pass

    def test_outside_away_bins_get_min_home_when_both_zero(self, occupancy_with_away, default_cluster_assumptions):
        """Bins outside the away window that are fully away get raised to one_unit home."""
        pass

    def test_none_away_day_leaves_states_unchanged(self, occupancy_always_home, default_cluster_assumptions):
        """When away_time for a day is None, no states are modified."""
        pass


# ---------------------------------------------------------------------------
# OccupancyGenerator — apply_sleep_time
# ---------------------------------------------------------------------------

class TestApplySleepTime:
    def test_sleep_bins_forced_to_all_sleep(self, occupancy_with_sleep, default_cluster_assumptions):
        """Bins inside sleep_time must have home=0, sleep=1.0."""
        pass

    def test_outside_sleep_bins_clamp_excess_sleep(self, occupancy_with_sleep, default_cluster_assumptions):
        """Bins outside sleep_time with sleep > max_awake_sleep_ratio should be clamped."""
        pass

    def test_none_sleep_day_leaves_states_unchanged(self, occupancy_with_sleep, default_cluster_assumptions):
        """When sleep_time for a day is None, states are not modified."""
        pass

    def test_excess_sleep_redistributed_to_home(self, occupancy_with_sleep, default_cluster_assumptions):
        """Clamped sleep fraction should be added back to home."""
        pass


# ---------------------------------------------------------------------------
# OccupancyGenerator — household_away_time_annually
# ---------------------------------------------------------------------------

class TestHouseholdAwayTimeAnnually:
    def test_returns_365_elements(self, occupancy_with_away, default_cluster_assumptions):
        """Should return exactly 365 elements."""
        occ = OccupancyGenerator(occupancy_with_away, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually()
        
        assert isinstance(away_times, list)
        assert len(away_times) == 365

    def test_always_occupied_yields_none(self, occupancy_always_home, default_cluster_assumptions):
        """If is_always_occupied=True, all days should be None."""
        occ = OccupancyGenerator(occupancy_always_home, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually()
        assert len(away_times) == 365
        assert all(at is None for at in away_times)

    def test_away_day (self, occupancy_with_away, default_cluster_assumptions):
        """Non-None elements should be TimeRange instances.
            Sampled away intervals should be centered near the specified away_interval mean (statistical)."""
        occ = OccupancyGenerator(occupancy_with_away, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually()
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

    def test_away_time_wraps_midnight(self, occupancy_with_away_night, default_cluster_assumptions):
        """Test that away_time intervals that wrap midnight are handled correctly."""
        occ = OccupancyGenerator(occupancy_with_away_night, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually()
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
    def test_returns_365_elements(self, occupancy_with_sleep, default_cluster_assumptions):
        pass

    def test_always_awake_yields_all_none(self, occupancy_always_home, default_cluster_assumptions):
        """If is_always_awake=True, all days should be None."""
        pass

    def test_sleep_elements_are_time_ranges_or_none(self, occupancy_with_sleep, default_cluster_assumptions):
        """Non-None elements should be TimeRange instances."""
        pass


# ---------------------------------------------------------------------------
# OccupancyGenerator — generate (integration)
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_no_patterns(self, occupancy_always_home, default_cluster_assumptions):
        """generate() with no patterns should return valid weekly structure."""
        pass

    def test_generate_with_away_pattern(self, occupancy_with_away, default_cluster_assumptions):
        """generate() with away pattern should enforce away windows."""
        pass

    def test_generate_with_sleep_pattern(self, occupancy_with_sleep, default_cluster_assumptions):
        """generate() with sleep pattern should enforce sleep windows."""
        pass

    def test_generate_with_both_patterns(self, occupancy_with_away_and_sleep, default_cluster_assumptions):
        """generate() with both patterns should apply both constraints."""
        pass

    def test_generate_output_fractions_valid(self, occupancy_always_home, default_cluster_assumptions):
        """All fractions in the output should satisfy 0 <= home + sleep <= 1."""
        pass

    def test_generate_output_week_count(self, occupancy_always_home, default_cluster_assumptions):
        """Output should contain 53 weeks covering 365 days."""
        pass


# ---------------------------------------------------------------------------
# OccupancyGenerator — static methods
# ---------------------------------------------------------------------------

class TestActiveSleepMask:
    def _make_states(self, home: float, sleep: float, num_weeks=1, bins_per_week=4) -> list[list[HouseholdOccupancyFractions]]:
        return [[HouseholdOccupancyFractions(home=home, sleep=sleep)] * bins_per_week for _ in range(num_weeks)]

    def test_all_away_produces_all_false_masks(self):
        """When all bins are fully away, both masks should be all False."""
        pass

    def test_high_home_ratio_flagged_as_active(self):
        """A bin with high home fraction should be active, not sleep."""
        pass

    def test_high_sleep_ratio_flagged_as_sleep(self):
        """A bin with high sleep fraction should be sleep, not active."""
        pass

    def test_threshold_boundary(self):
        """A bin exactly at the active_threshold should be classified as active."""
        pass

    def test_masks_are_mutually_exclusive(self):
        """A bin cannot be both active and sleep simultaneously."""
        pass

    def test_output_shape_matches_input(self):
        """active_mask and sleep_mask must have the same shape as occ_states."""
        pass


class TestApplyToMask:
    def test_masked_bins_get_value(self):
        """Bins where mask=True should receive the specified value."""
        pass

    def test_unmasked_bins_unchanged(self):
        """Bins where mask=False should retain their original value."""
        pass

    def test_mismatched_lengths_raise(self):
        """Passing mask and schedule with different lengths should raise AssertionError."""
        pass

    def test_all_false_mask_returns_original(self):
        """An all-False mask should return the schedule unchanged."""
        pass

    def test_all_true_mask_returns_all_value(self):
        """An all-True mask should replace every element with value."""
        pass


class TestToOccupancySchedule:
    def test_length_equals_365_days(self):
        """Flattened schedule length should equal 365 * bins_per_day."""
        pass

    def test_values_are_home_plus_sleep(self):
        """Each value should equal the corresponding home + sleep fraction."""
        pass

    def test_values_in_zero_one_range(self):
        """All schedule values should be in [0.0, 1.0]."""
        pass

    def test_fully_away_state_yields_zero(self):
        """A fully-away bin should contribute 0.0 to the schedule."""
        pass

    def test_fully_home_state_yields_one(self):
        """A fully-home bin (home=1.0, sleep=0.0) should contribute 1.0."""
        pass
