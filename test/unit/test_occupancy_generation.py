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
    OccupancyState,
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
# OccupancyGenerator — household_away_time_annually
# ---------------------------------------------------------------------------

class TestHouseholdAwayTimeAnnually:
    def test_returns_365_elements(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
        """Should return exactly 365 elements."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually()
        
        assert isinstance(away_times, list)
        assert len(away_times) == 365

    def test_always_occupied_yields_none(self, occupancy_always_home, default_cluster_assumptions):
        """If is_always_occupied=True, all days should be None."""
        occ = OccupancyGenerator(occupancy_always_home, default_cluster_assumptions, SIM_RES)
        away_times = occ.household_away_time_annually()
        assert len(away_times) == 365
        assert all(at is None for at in away_times)

    def test_away_day (self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
        """Non-None elements should be TimeRange instances.
            Sampled away intervals should be centered near the specified away_interval mean (statistical)."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
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

    @pytest.mark.parametrize("rigidness, expected_std", [
        (ScheduleRigidness.STRICT,            0.25),
        (ScheduleRigidness.MOSTLY_CONSISTENT, 0.5),
        (ScheduleRigidness.SOMEWHAT_VARIABLE, 1.0),
        (ScheduleRigidness.HIGHLY_VARIABLE,   2.0),
    ])
    def test_away_time_variance_matches_rigidness(self, default_cluster_assumptions, rigidness, expected_std):
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
        away_times = occ.household_away_time_annually()

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
    def test_returns_with_away_times(self, occupancy_with_away_and_sleep, default_cluster_assumptions):
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        away_time = occ.household_away_time_annually()
        sleep_time = occ.household_sleep_time_annually(away_time)
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


    def test_always_awake_yields_all_none(self, occupancy_always_home, default_cluster_assumptions):
        """If is_always_awake=True, all days should be None."""
        occ = OccupancyGenerator(occupancy_always_home, default_cluster_assumptions, SIM_RES)
        away_time = occ.household_away_time_annually()
        sleep_time = occ.household_sleep_time_annually(away_time)
        assert len(sleep_time) == 365
        assert all(st is None for st in sleep_time)

    @pytest.mark.parametrize("rigidness, expected_std", [
        (ScheduleRigidness.STRICT,            0.25),
        (ScheduleRigidness.MOSTLY_CONSISTENT, 0.5),
        (ScheduleRigidness.SOMEWHAT_VARIABLE, 1.0),
        (ScheduleRigidness.HIGHLY_VARIABLE,   2.0),
    ])
    def test_sleep_time_variance_matches_rigidness(self, default_cluster_assumptions, rigidness, expected_std):
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
        sleep_times = occ.household_sleep_time_annually([None] * 365)

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

    def test_sleep_no_overlap_with_away(self, occupancy_with_away_and_sleep_overlap, default_cluster_assumptions):
        """Non-None elements should be TimeRange instances."""
        occ = OccupancyGenerator(occupancy_with_away_and_sleep_overlap, default_cluster_assumptions, SIM_RES)
        away_time = occ.household_away_time_annually()
        sleep_time = occ.household_sleep_time_annually(away_time)
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
    def test_returns(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        states = occ.household_mc_state_annually()
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

    def test_away_bins_forced_to_zero(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
        """Bins inside the away_time window must have home=0, sleep=0."""
        occ = OccupancyGenerator(occupancy_with_away_cluster_long_day_away, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0

        # Fixed (no-variance) away intervals for deterministic checking
        wd_away = TimeRange(start_hour=8, end_hour=18)
        we_away = TimeRange(start_hour=16, end_hour=20)
        away_time = [we_away if (d % 7) in [0, 6] else wd_away for d in range(365)]

        states = occ.apply_away_time(occ.household_mc_state_annually(), away_time)

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

    def test_average_away_bins_matches_interval(self, occupancy_with_away_cluster_long_day_away, default_cluster_assumptions):
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

        away_time = occ.household_away_time_annually()
        states = occ.apply_away_time(occ.household_mc_state_annually(), away_time)

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

    def test_sleep_bins_forced_to_all_sleep(self, occupancy_with_away_and_sleep, default_cluster_assumptions):
        """Bins inside sleep_time must have home=0, sleep=1.0."""
        occ = OccupancyGenerator(occupancy_with_away_and_sleep, default_cluster_assumptions, SIM_RES)
        bins_per_day = 1440 // SIM_RES
        resolution_hours = SIM_RES / 60.0

        sleep_range = TimeRange(start_hour=22, end_hour=6)  # fixed, no variance
        sleep_time = [sleep_range] * 365

        states = occ.apply_sleep_time(occ.household_mc_state_annually(), sleep_time)

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
        result = occ.apply_sleep_time(initial_states, sleep_time, max_awake_sleep_ratio=0.0)

        for day in range(365):
            week, day_in_week = day // 7, day % 7
            day_start = day_in_week * bins_per_day
            for b in range(bins_per_day):
                f = result[week][day_start + b]
                if not sleep_range.contains_hour(b * resolution_hours):
                    assert f.sleep == pytest.approx(0.0), (
                        f"Day {day} bin {b}: sleep not clamped (sleep={f.sleep})"
                    )
                    assert f.home == pytest.approx(initial_home + initial_sleep), (
                        f"Day {day} bin {b}: excess sleep not redistributed to home (home={f.home})"
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

    def test_average_sleep_bins_matches_interval(self, occupancy_with_away_and_sleep, default_cluster_assumptions):
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

        sleep_time = occ.household_sleep_time_annually([None] * 365)
        states = occ.apply_sleep_time(occ.household_mc_state_annually(), sleep_time, max_awake_sleep_ratio=0.0)

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
    def test_generate_3_people(self, default_cluster_assumptions):
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
        schedule = occ_gen.generate()
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
