import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from collections.abc import Sequence
from matplotlib.figure import Figure
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.generator import occupancy_generator


def get_project_root() -> Path:
    """Find the project root directory.

    Tries multiple strategies:
    1. Checks OCCUPANCY_GENERATION_ROOT environment variable
    2. Uses __file__ if available (works in normal execution)
    3. Searches upward from current working directory for pyproject.toml or data directory
    4. Fallback using 'src' in the current working directory path

    Returns:
        Path to project root directory
    """
    # Strategy 1: Check environment variable
    env_root = os.getenv("OCCUPANCY_GENERATION_ROOT")
    if env_root:
        root = Path(env_root)
        if root.exists() and (root / "pyproject.toml").exists():
            return root
    
    # Strategy 2: Try __file__ approach (works when running as script/module)
    try:
        # This file is at src/obgeneration/generator/ob_utils.py, so go up 4 levels to project root
        file_path = Path(__file__).resolve()
        if file_path.exists():
            # src/obgeneration/generator/ob_utils.py -> generator -> obgeneration -> src -> project_root
            candidate = file_path.parent.parent.parent.parent
            if (candidate / "pyproject.toml").exists() or (candidate / "data").exists():
                return candidate
    except (NameError, AttributeError):
        pass  # __file__ not available (e.g., in some REPLs)
    
    # Strategy 3: Search upward from current working directory
    cwd = Path.cwd().resolve()
    current = cwd
    for _ in range(10):  # Limit search depth
        if (current / "pyproject.toml").exists() or (current / "data").exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Strategy 4: Fallback - assume we're in src/generator and go up 2 levels
    # This is a last resort
    fallback = Path.cwd()
    if "src" in str(fallback):
        # Find src directory and go up one level
        parts = fallback.parts
        if "src" in parts:
            idx = parts.index("src")
            return Path(*parts[:idx])
    
    # Final fallback: return current directory (may not be correct, but won't crash)
    return Path.cwd()


class ScheduleUtils:
    @staticmethod
    def weekly_index_day(index: int, resolution_mins: int) -> int:
        """ Returns the day of the week for a given index in a weekly schedule. """
        intervals_per_day = int(1440 / resolution_mins)
        return (index // intervals_per_day) % 7
    
    @staticmethod
    def weekly_index_hour(index: int, resolution_mins: int) -> float:
        """ Returns the hour of the day for a given index in a weekly schedule. """
        intervals_per_day = int(1440 / resolution_mins)
        index_in_day = index % intervals_per_day
        return (index_in_day * resolution_mins) / 60.0  
    
    @staticmethod
    def get_day_indices(day: int, resolution_mins: int) -> list[int]:
        """ Returns the list of indices for a specific day in a weekly schedule. """
        intervals_per_day = int(1440 / resolution_mins)
        start_index = day * intervals_per_day
        return list(range(start_index, start_index + intervals_per_day))

    @staticmethod
    def flatten_schedule(schedule):
        """ Flattens a schedule of lists into a single list. """
        return [item for sublist in schedule for item in sublist]
    
    # @staticmethod
    # def get_typical_week_values_from_occ(schedule: list[list[HouseholdOccupancyFractions]]) -> np.ndarray:
    #     occ_schedule = occupancy_generator.OccupancyGenerator.to_occupancy_schedule(schedule)
    #     avg_week = ScheduleUtils.get_typical_week_values(occ_schedule)
    #     return avg_week


    @staticmethod
    def get_typical_week_values(schedule: Sequence[float] | Sequence[Sequence[float]]) -> tuple[np.ndarray, int, int, float]:
        """ Calculates the average schedule for a typical week by averaging the first 52 weeks of data.
        """
       #if nested schedule, flatten it
        if schedule and isinstance(schedule[0], (Sequence, np.ndarray)) and not isinstance(schedule[0], (str, bytes)):
            schedule = ScheduleUtils.flatten_schedule(schedule)
        timesteps_per_day = len(schedule) // 365
        week_len = 7 * timesteps_per_day

        first_52_weeks = schedule[:52 * week_len]
        weeks_reshaped = np.array(first_52_weeks, dtype=float).reshape(52, week_len)
        avg_week = weeks_reshaped.mean(axis=0)
        return avg_week

        


    @staticmethod
    def annual_schedule_heatmap_figure(
        flat_schedule: Sequence[float],
        title: str,
        colormap: str = "YlGn",
        vmin=None,
        vmax=None,
    ) -> Figure:
        """Plot a sequential flat annual schedule as a 365-day heatmap.

        Args:
            flat_schedule: Flat array-like schedule ordered sequentially in time across the year.
            title: Figure title.
            colormap: Matplotlib colormap name.
            vmin: Optional lower bound for the color scale.
            vmax: Optional upper bound for the color scale.

        Returns:
            matplotlib.figure.Figure: Figure containing the heatmap.
        """
        if len(flat_schedule) == 0:
            raise ValueError("flat_schedule must not be empty.")
        if len(flat_schedule) % 365 != 0:
            raise ValueError(
                f"flat_schedule length {len(flat_schedule)} is not divisible by 365."
            )

        timesteps_per_day = len(flat_schedule) // 365
        minutes_per_step = 1440 / timesteps_per_day
        steps_per_hour = 60 / minutes_per_step
        heatmap_data = np.array(flat_schedule, dtype=float).reshape(365, timesteps_per_day).T

        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )

        colorbar = fig.colorbar(im, ax=ax)
        colorbar.set_label("Schedule", rotation=270, labelpad=20)

        week_ticks = [day for day in range(0, 365, 28)]
        ax.set_xticks(week_ticks)
        ax.set_xticklabels([f"Week {day // 7}" for day in week_ticks])
        ax.set_xlabel("Day of Year")

        hour_tick_positions = [
            int(hour * steps_per_hour)
            for hour in range(0, 25, 2)
            if int(hour * steps_per_hour) < timesteps_per_day
        ]
        ax.set_yticks(hour_tick_positions)
        ax.set_yticklabels([
            str(hour)
            for hour in range(0, 25, 2)
            if int(hour * steps_per_hour) < timesteps_per_day
        ])
        ax.set_ylabel("Hour of Day")
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_annual_schedule_heatmap(schedule: Sequence[float] | Sequence[Sequence[float]], title: str,  colormap: str = 'YlGn', vmin=0, vmax=1) -> None:
        """Plot annual schedule as a heatmap with days on x-axis and timesteps per day on y-axis."""
        if schedule and isinstance(schedule[0], (Sequence, np.ndarray)) and not isinstance(schedule[0], (str, bytes)):
            schedule = ScheduleUtils.flatten_schedule(schedule)
        return ScheduleUtils.annual_schedule_heatmap_figure(
            flat_schedule=schedule,
            title=title,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
        )


    @staticmethod
    def plot_typical_week_schedule(typical_week: np.array, title: str,  label: str = 'Schedule', vmin=0, vmax=1) -> None:
        """Plot typical week schedule by averaging the first 52 weeks.
        
        Args:
            schedule: Either a flattened list of floats or nested list of lists (weeks)
            title: Title for the plot
            out_path: Path where the plot will be saved
            resolution_mins: Time resolution in minutes (used to calculate timesteps per day)
            label: Label for the plot line (default: 'Schedule')
        """
        week_len = len(typical_week)
        timesteps_per_day = week_len // 7
        num_per_hour = 60 / (1440 / timesteps_per_day)
        # ---- plot ----
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, typical_week, label=label, linewidth=1.5)

        # Major ticks: day labels at center of each day
        entries_per_day = timesteps_per_day
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Sun","Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
        ax.set_xlim(0, week_len)

        # Minor ticks: hour markers at 0, 6, 12, 18 for each day with labels
        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + int(hour * num_per_hour))
                hour_labels.append(str(hour))

        ax2 = ax.twiny()  # Create secondary x-axis for hour labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        # Add vertical grid lines at hour markers
        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)
    

        ax.set_title(title)
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')

        ax.set_ylim(vmin, vmax)
        # Handle path
        return fig
        

    @staticmethod
    def typical_week_schedule_figure(schedule: Sequence[float] | Sequence[Sequence[float]], title: str, resolution_mins: int, label: str = 'Schedule', vmin=0, vmax=1) -> Figure:
        """Plot typical week schedule by averaging the first 52 weeks.
        
        Returns a Figure that can be displayed in a notebook (e.g., with plt.show() or
        as the last expression in a cell).
        
        Args:
            schedule: Either a flattened list of floats or nested list of lists (weeks)
            title: Title for the plot
            resolution_mins: Time resolution in minutes (used to calculate timesteps per day)
            label: Label for the plot line (default: 'Schedule')
            vmin: Y-axis minimum (default: 0)
            vmax: Y-axis maximum (default: 1)
            
        Returns:
            matplotlib.figure.Figure: A figure that can be shown with plt.show(fig) or displayed in a notebook.
        """
        avg_week, timesteps_per_day, week_len, num_per_hour = ScheduleUtils.get_typical_week_values(schedule, resolution_mins)

        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week, label=label, linewidth=1.5)

        entries_per_day = timesteps_per_day
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Sun","Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
        ax.set_xlim(0, week_len)

        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + int(hour * num_per_hour))
                hour_labels.append(str(hour))

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)

        ax.set_title(title)
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')
        ax.set_ylim(vmin, vmax)
        fig.tight_layout()
        return fig

    @staticmethod
    def single_day_schedule_figure(schedule: Sequence[float] | Sequence[Sequence[float]], day: int, title: str, resolution_mins: int, label: str = 'Schedule', vmin=0, vmax=1) -> Figure:
        """Plot a single day's schedule (averaged across first 52 weeks for that weekday).

        Day is 0=Monday through 6=Sunday. Style matches proc.ipynb: hours 1-24 on x-axis,
        markers, grid. Returns a Figure for display in a notebook.

        Args:
            schedule: Either a flattened list of floats or nested list of lists (weeks)
            day: Day of week to plot (0=Mon, 1=Tue, ..., 6=Sun)
            title: Title for the plot
            resolution_mins: Time resolution in minutes
            label: Label for the plot line (default: 'Schedule')
            vmin: Y-axis minimum (default: 0)
            vmax: Y-axis maximum (default: 1)

        Returns:
            matplotlib.figure.Figure: A figure that can be shown with plt.show(fig) or displayed in a notebook.
        """
        avg_week, timesteps_per_day, _, _ = ScheduleUtils.get_typical_week_values(schedule, resolution_mins)
        day = max(0, min(6, day))
        day_data = avg_week[day * timesteps_per_day : (day + 1) * timesteps_per_day]

        # Aggregate to hourly (24 values) to match proc.ipynb style
        steps_per_hour = timesteps_per_day // 24
        hourly = np.array([day_data[i * steps_per_hour : (i + 1) * steps_per_hour].mean() for i in range(24)])

        hours = np.arange(1, 25)
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        ax.plot(hours, hourly, marker="o", markersize=4, label=label)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Fraction")
        ax.set_title(title)
        ax.set_xticks(hours)
        ax.set_xticklabels(hours)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_two_typical_week_schedule(schedule1: Sequence[float] | Sequence[Sequence[float]], schedule2: Sequence[float] | Sequence[Sequence[float]], title: str, out_path: str | Path, resolution_mins: int, label1: str = 'Schedule 1', label2: str = 'Schedule 2') -> None:
        """Plot typical week schedule by averaging the first 52 weeks.
        
        Args:
            schedule1: First schedule (flattened or nested)
            schedule2: Second schedule (flattened or nested)
            title: Title for the plot
            out_path: Path where the plot will be saved
            resolution_mins: Time resolution in minutes (used to calculate timesteps per day)
            label1: Label for the first schedule (default: 'Schedule 1')
            label2: Label for the second schedule (default: 'Schedule 2')
        """
        avg_week1, timesteps_per_day, week_len, num_per_hour = ScheduleUtils.get_typical_week_values(schedule1, resolution_mins)
        avg_week2, _, _, _ = ScheduleUtils.get_typical_week_values(schedule2, resolution_mins)
        avg_weeks = [avg_week1, avg_week2]

        # ---- plot ----
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_weeks[0], label=label1, linewidth=1.5)
        ax.plot(x, avg_weeks[1], label=label2, linewidth=1.5)

        # Major ticks: day labels at center of each day
        entries_per_day = timesteps_per_day
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Sun","Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
        ax.set_xlim(0, week_len)

        # Minor ticks: hour markers at 0, 6, 12, 18 for each day with labels
        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + int(hour * num_per_hour))
                hour_labels.append(str(hour))

        ax2 = ax.twiny()  # Create secondary x-axis for hour labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        # Add vertical grid lines at hour markers
        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)
    

        ax.set_title(title)
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')

        vmin = min(np.min(avg_weeks[0]), np.min(avg_weeks[1]))
        vmax = max(np.max(avg_weeks[0]), np.max(avg_weeks[1]))
        offset = min(1,vmin)
        ax.set_ylim(vmin - 0.1 * offset, vmax + 0.1 * abs(vmax))
        # Handle path
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
