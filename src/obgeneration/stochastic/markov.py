"""Markov chain occupancy state sampling."""

from enum import Enum
from importlib.resources import files

import numpy as np
import pandas as pd


class OccupancyState(Enum):
    AWAY = 0
    HOME = 1
    SLEEP = 2


class ClusterAssumptions:
    """
    Stores Markov-chain parameters for occupancy state simulation.

    States are defined by OccupancyState:
        AWAY  = 0
        HOME  = 1
        SLEEP = 2

    The day is divided into (1440 / resolution_min) bins, starting at start_min.
    With the defaults (start_min=240, resolution_min=15):
        bin 0  → 04:00–04:15
        bin 1  → 04:15–04:30
        ...
        bin 95 → 03:45–04:00 (next day)

    Attributes
    ----------
    num_clusters : int
        Number of household behaviour clusters.
    start_min : int
        The minute-of-day at which bin 0 begins (e.g. 240 = 04:00).
    assumption_resolution_min : int
        Resolution of the loaded transition matrices in minutes.
    weekday_initial_probs : list[list[float]], shape [num_clusters, 3]
    weekend_initial_probs : list[list[float]], shape [num_clusters, 3]
    weekday_transition_probs : dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]}
    weekend_transition_probs : dict[int, list[list[list[float]]]], shape {cluster: [num_bins, 3, 3]}
    """

    _STATE_FROM_ORDER = {
        "Away": OccupancyState.AWAY.value,
        "Home": OccupancyState.HOME.value,
        "Sleep": OccupancyState.SLEEP.value,
    }
    _TO_COLS = ["to_Away", "to_Home", "to_Sleep"]

    CLUSTER_INDEX = {
        "mostly_home": 0,
        "long_day_away": 1,
        "morning_away": 2,
        "afternoon_away": 3,
        "evening_night_away": 4,
    }

    def __init__(
        self,
        num_clusters: int,
        start_min: int,
        assumption_resolution_min: int,
        weekday_initial_probs: list[list[float]],
        weekend_initial_probs: list[list[float]],
        weekday_transition_probs: dict[int, list[list[list[float]]]],
        weekend_transition_probs: dict[int, list[list[list[float]]]],
    ):
        self.num_clusters = num_clusters
        self.start_min = start_min
        self.assumption_resolution_min = assumption_resolution_min
        self.weekday_initial_probs = weekday_initial_probs
        self.weekend_initial_probs = weekend_initial_probs
        self.weekday_transition_probs = weekday_transition_probs
        self.weekend_transition_probs = weekend_transition_probs
        # Precompute cumulative sums for fast searchsorted sampling
        self._wd_init_cumsum = np.cumsum(np.array(weekday_initial_probs), axis=-1)
        self._we_init_cumsum = np.cumsum(np.array(weekend_initial_probs), axis=-1)
        self._wd_cumsum = {k: np.cumsum(np.array(v), axis=-1) for k, v in weekday_transition_probs.items()}
        self._we_cumsum = {k: np.cumsum(np.array(v), axis=-1) for k, v in weekend_transition_probs.items()}

    @staticmethod
    def _read_transition_probs(path) -> list[list[list[float]]]:
        df = pd.read_csv(path)
        df["from_idx"] = df["from_state"].map(ClusterAssumptions._STATE_FROM_ORDER)
        df = df.sort_values(["bin", "from_idx"])
        return [
            group[ClusterAssumptions._TO_COLS].values.tolist()
            for _, group in df.groupby("bin")
        ]

    @classmethod
    def default(cls) -> "ClusterAssumptions":
        data_dir = files("obgeneration.data.occupancy_probability")
        num_clusters = 5
        df = pd.read_csv(data_dir / "weekday" / "weekday_pi0.csv").sort_values("raw_cluster")
        weekday_initial_probs = df[["Away", "Home", "Sleep"]].values.tolist()
        df = pd.read_csv(data_dir / "weekend" / "weekend_pi0.csv").sort_values("raw_cluster")
        weekend_initial_probs = df[["Away", "Home", "Sleep"]].values.tolist()
        weekday_transition_probs = {
            i: cls._read_transition_probs(data_dir / "weekday" / f"cluster_{i+1:02d}_transitions.csv")
            for i in range(num_clusters)
        }
        weekend_transition_probs = {
            i: cls._read_transition_probs(data_dir / "weekend" / f"cluster_{i+1:02d}_transitions.csv")
            for i in range(num_clusters)
        }
        return cls(num_clusters, 240, 15, weekday_initial_probs, weekend_initial_probs, weekday_transition_probs, weekend_transition_probs)

    def _build_cumsum(self, transition_matrices: np.ndarray, resolution_min: int) -> np.ndarray:
        if resolution_min != self.assumption_resolution_min:
            raise ValueError(
                f"Markov transitions are only defined at the native assumption resolution "
                f"of {self.assumption_resolution_min} minutes, got {resolution_min}."
            )
        return np.cumsum(transition_matrices, axis=-1)

    def sample_cluster_annually(
        self,
        weekday_cluster_id: int,
        weekend_cluster_id: int,
        sim_resolution_min: int,
        rng: np.random.Generator,
    ) -> list[list[OccupancyState]]:
        """Simulate a full year of occupancy states for one occupant.

        Returns 53 weeks (52 full + 1 partial Saturday), each a flat list of
        OccupancyState values starting at Sunday midnight.
        """
        if sim_resolution_min != self.assumption_resolution_min:
            raise ValueError(
                f"sample_cluster_annually only supports the native assumption resolution "
                f"of {self.assumption_resolution_min} minutes, got {sim_resolution_min}."
            )
        num_bins = 1440 // sim_resolution_min
        midnight_bin = (1440 - self.start_min) // sim_resolution_min
        we_cumsum = self._build_cumsum(self._we_cumsum[weekend_cluster_id], sim_resolution_min)
        wd_cumsum = self._build_cumsum(self._wd_cumsum[weekday_cluster_id], sim_resolution_min)

        total_steps = 1 + 366 * num_bins
        randoms = rng.random(total_steps)
        r = 0

        current_state = int(np.searchsorted(self._we_init_cumsum[weekend_cluster_id], randoms[r])); r += 1

        initial_state = []
        for b in range(num_bins):
            current_state = int(np.searchsorted(we_cumsum[b, current_state], randoms[r])); r += 1
            if b >= midnight_bin:
                initial_state.append(current_state)

        states = np.empty(366 * num_bins - midnight_bin, dtype=np.int8)
        states[:(num_bins - midnight_bin)] = initial_state
        idx = num_bins - midnight_bin

        for day in range(365):
            cumsum = we_cumsum if (day % 7) in [0, 6] else wd_cumsum
            for b in range(num_bins):
                current_state = int(np.searchsorted(cumsum[b, current_state], randoms[r])); r += 1
                states[idx] = current_state; idx += 1

        flat = [OccupancyState(s) for s in states[:365 * num_bins]]
        bins_per_week = 7 * num_bins
        return [flat[i:i + bins_per_week] for i in range(0, len(flat), bins_per_week)]
