from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from obgeneration.generator.occupancy_generator import ClusterAssumptions


SIM_RES = 15


@pytest.fixture(scope="module")
def default_cluster_assumptions() -> ClusterAssumptions:
    return ClusterAssumptions.default()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


class TestClusterAssumptions:
    def test_default_loads(self, default_cluster_assumptions):
        ca = default_cluster_assumptions
        assert ca.num_clusters == 5
        assert len(ca.weekday_initial_probs) == 5
        assert all(len(row) == 3 for row in ca.weekday_initial_probs)
        assert len(ca.weekend_initial_probs) == 5
        assert all(len(row) == 3 for row in ca.weekend_initial_probs)
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
        ca = default_cluster_assumptions
        num_bins = 1440 // ca.assumption_resolution_min
        matrix = np.array(ca.weekday_transition_probs[0])
        result = ca._build_cumsum(matrix, ca.assumption_resolution_min)
        assert result.shape == (num_bins, 3, 3)

    def test_upsample_doubles_bins(self):
        assumption_res = 30
        num_bins = 1440 // assumption_res
        uniform_row = [1 / 3, 1 / 3, 1 / 3]
        trans_probs = {i: [[uniform_row, uniform_row, uniform_row]] * num_bins for i in range(5)}
        init_probs = [[1 / 3, 1 / 3, 1 / 3]] * 5
        ca = ClusterAssumptions(5, 240, assumption_res, init_probs, init_probs, trans_probs, trans_probs)
        matrix = np.array(trans_probs[0])
        result = ca._build_cumsum(matrix, assumption_res // 2)
        assert result.shape[0] == num_bins * 2

    def test_downsample_halves_bins(self, default_cluster_assumptions):
        ca = default_cluster_assumptions
        num_bins = 1440 // ca.assumption_resolution_min
        matrix = np.array(ca.weekday_transition_probs[0])
        result = ca._build_cumsum(matrix, ca.assumption_resolution_min * 2)
        assert result.shape[0] == num_bins // 2

    def test_cumsum_last_value_is_one(self, default_cluster_assumptions):
        ca = default_cluster_assumptions
        matrix = np.array(ca.weekday_transition_probs[0])
        result = ca._build_cumsum(matrix, ca.assumption_resolution_min)
        assert result[:, :, -1] == pytest.approx(np.ones((result.shape[0], 3)))


class TestSampleCluster:
    def test_returns_53_weeks(self, default_cluster_assumptions, rng):
        weeks = default_cluster_assumptions.sample_cluster_annually(0, 0, SIM_RES, rng)
        assert isinstance(weeks, list)
        assert len(weeks) == 53
        for w in range(53):
            week = weeks[w]
            assert isinstance(week, list)
            if w < 52:
                assert len(week) == 7 * (1440 // SIM_RES)
            else:
                assert len(week) == 1 * (1440 // SIM_RES)

    def test_output_plot(self, default_cluster_assumptions, rng):
        matplotlib.use("Agg")

        output_dir = Path(__file__).parents[0] / "output"
        output_dir.mkdir(exist_ok=True)

        bins_per_day = 1440 // SIM_RES
        state_labels = {0: "Away", 1: "Home", 2: "Sleep"}
        cmap = matplotlib.colors.ListedColormap(["#d62728", "#2ca02c", "#1f77b4"])
        norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        y_ticks = list(range(0, bins_per_day, bins_per_day // 8))
        y_labels = [f"{int(t * SIM_RES / 60):02d}:00" for t in y_ticks]

        cluster_names = ["mostly_home", "long_day_away", "morning_away", "afternoon_away", "evening_night_away"]

        for cluster in range(5):
            weeks = default_cluster_assumptions.sample_cluster_annually(cluster, cluster, SIM_RES, rng)
            flat = [s.value for week in weeks for s in week][:365 * bins_per_day]
            grid = np.array(flat, dtype=np.int8).reshape(365, bins_per_day).T

            fig, ax = plt.subplots(figsize=(18, 6))
            im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, origin="upper", extent=[0, 365, bins_per_day, 0])

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

    def test_average_occupancy_line_plot(self, default_cluster_assumptions, rng):
        matplotlib.use("Agg")

        output_dir = Path(__file__).parents[0] / "output"
        output_dir.mkdir(exist_ok=True)

        bins_per_day = 1440 // SIM_RES
        cluster_names = ["mostly_home", "long_day_away", "morning_away", "afternoon_away", "evening_night_away"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        x_ticks = list(range(0, bins_per_day, bins_per_day // 8))
        x_labels = [f"{int(t * SIM_RES / 60):02d}:00" for t in x_ticks]

        day_indices = np.arange(365)
        weekday_mask = ~np.isin(day_indices % 7, [0, 6])
        weekend_mask = np.isin(day_indices % 7, [0, 6])

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        fig.suptitle("Average Occupied Fraction by Time of Day (Home + Sleep)")

        for cluster in range(5):
            weeks = default_cluster_assumptions.sample_cluster_annually(cluster, cluster, SIM_RES, rng)
            flat = np.array([s.value for week in weeks for s in week][:365 * bins_per_day], dtype=np.int8)
            grid = flat.reshape(365, bins_per_day)
            occupied = (grid >= 1).astype(float)

            wd_avg = occupied[weekday_mask].mean(axis=0)
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
