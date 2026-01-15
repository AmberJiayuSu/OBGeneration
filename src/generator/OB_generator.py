
from stochastic.distribution import Distribution
from model.occupant_profile import Occupant
from generator.occupancy_generator import OccupancyGenerator, OccupancyAssumptions
from generator.lighting_generator import LightingGenerator
from generator.equipment_generator import EquipmentGenerator, EquipmentAssumptions
from generator.dhw_generator import DHWGenerator, DHWAssumptions
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

import matplotlib.pyplot as plt
import numpy as np


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

    @staticmethod
    def plot_annual_schedule_heatmap(schedule: list[float] | list[list[float]], title: str, out_path: str | Path, resolution_mins: int, colormap: str = 'YlGn') -> None:
        """Plot annual schedule as a heatmap with days on x-axis and timesteps per day on y-axis.
        
        Args:
            schedule: Either a flattened list of floats or nested list of lists (weeks)
            title: Title for the plot
            out_path: Path where the plot will be saved
            resolution_mins: Time resolution in minutes (used to calculate timesteps per day)
            colormap: Colormap name for the heatmap (default: 'YlGn')
        """
        # Calculate timesteps
        num_per_hour = 60 / resolution_mins
        timesteps_per_day = int(24 * 60 / resolution_mins)
        
        # Detect if schedule is flattened or nested
        if schedule and isinstance(schedule[0], list):
            # Nested schedule (list of weeks)
            all_days = []
            for week_idx, week_data in enumerate(schedule):
                week_array = np.array(week_data)
                if week_idx < 52:
                    # Full week: reshape into 7 days
                    for day in range(7):
                        day_data = week_array[day * timesteps_per_day:(day + 1) * timesteps_per_day]
                        all_days.append(day_data)
                else:
                    # Last partial week (1 day)
                    all_days.append(week_array[:timesteps_per_day])
            
            # Stack into 2D array: rows = time slots, cols = days
            heatmap_data = np.column_stack(all_days)  # Shape: (timesteps_per_day, 365)
        else:
            # Flattened schedule: reshape into 2D array
            schedule_array = np.array(schedule)
            num_days = len(schedule_array) // timesteps_per_day
            # Reshape: (timesteps_per_day, num_days)
            heatmap_data = schedule_array[:num_days * timesteps_per_day].reshape(timesteps_per_day, num_days)

        fig2, ax_heat = plt.subplots(figsize=(12, 3))

        # Create heatmap
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap=colormap, vmin=0, vmax=1, origin='upper')

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax_heat)
        cbar.set_label('Schedule', rotation=270, labelpad=20)

        # Set x-axis (days)
        num_days = heatmap_data.shape[1]
        # Show ticks at week boundaries (every 7 days)
        week_ticks = [w * 7 for w in range(0, (num_days // 7) + 1, 4) if w * 7 < num_days]
        ax_heat.set_xticks(week_ticks)
        ax_heat.set_xticklabels([f'Week {w}' for w in range(0, (num_days // 7) + 1, 4) if w * 7 < num_days])
        ax_heat.set_xlabel('Day of Year')

        # Set y-axis (hours of day)
        # Show hour labels at every 2 hours
        hour_tick_positions = [int(h * num_per_hour) for h in range(0, 25, 2) if int(h * num_per_hour) < timesteps_per_day]
        ax_heat.set_yticks(hour_tick_positions)
        ax_heat.set_yticklabels([str(h) for h in range(0, 25, 2) if int(h * num_per_hour) < timesteps_per_day])
        ax_heat.set_ylabel('Hour of Day')


        ax_heat.set_title(title)

        fig2.tight_layout()
        
        # Handle path
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out_path, dpi=200)
        plt.close(fig2)


    @staticmethod
    def plot_typical_week_schedule(schedule: list[float] | list[list[float]], title: str, out_path: str | Path, resolution_mins: int, label: str = 'Schedule') -> None:
        """Plot typical week schedule by averaging the first 52 weeks.
        
        Args:
            schedule: Either a flattened list of floats or nested list of lists (weeks)
            title: Title for the plot
            out_path: Path where the plot will be saved
            resolution_mins: Time resolution in minutes (used to calculate timesteps per day)
            label: Label for the plot line (default: 'Schedule')
        """
        # Calculate timesteps
        num_per_hour = 60 / resolution_mins
        timesteps_per_day = int(24 * 60 / resolution_mins)
        week_len = 7 * timesteps_per_day
        
        # Detect if schedule is flattened or nested and extract first 52 weeks
        if schedule and isinstance(schedule[0], list):
            # Nested schedule (list of weeks)
            # Take first 52 weeks
            weeks_to_average = schedule[:52]
            # Convert to numpy array: shape (52, week_len)
            W = np.array([week_data[:week_len] for week_data in weeks_to_average], dtype=float)
        else:
            # Flattened schedule: reshape into weeks, then take first 52
            schedule_array = np.array(schedule)
            num_weeks = len(schedule_array) // week_len
            # Reshape: (num_weeks, week_len), then take first 52
            weeks_reshaped = schedule_array[:num_weeks * week_len].reshape(num_weeks, week_len)
            W = weeks_reshaped[:52]  # shape (52, week_len) or less if not enough weeks
        
        # Average across weeks
        avg_week = W.mean(axis=0)  # shape (week_len,)

        # ---- plot ----
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week, label=label, linewidth=1.5)

        # Major ticks: day labels at center of each day
        entries_per_day = timesteps_per_day
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
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

        ax.set_ylim(-0.1, 1.1)

        # Handle path
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    @staticmethod
    def plot_two_typical_week_schedule(schedule1: list[float] | list[list[float]], schedule2: list[float] | list[list[float]], title: str, out_path: str | Path, resolution_mins: int, label1: str = 'Schedule 1', label2: str = 'Schedule 2') -> None:
        """Plot typical week schedule by averaging the first 52 weeks.
        
        Args:
            schedule: Either a flattened list of floats or nested list of lists (weeks)
            title: Title for the plot
            out_path: Path where the plot will be saved
            resolution_mins: Time resolution in minutes (used to calculate timesteps per day)
            label: Label for the plot line (default: 'Schedule')
        """
        # Calculate timesteps
        num_per_hour = 60 / resolution_mins
        timesteps_per_day = int(24 * 60 / resolution_mins)
        week_len = 7 * timesteps_per_day
        avg_weeks = []
        
        for schedule in [schedule1, schedule2]:
            # Detect if schedule is flattened or nested and extract first 52 weeks
            if schedule and isinstance(schedule[0], list):
                # Nested schedule (list of weeks)
                # Take first 52 weeks
                weeks_to_average = schedule[:52]
                # Convert to numpy array: shape (52, week_len)
                W = np.array([week_data[:week_len] for week_data in weeks_to_average], dtype=float)
            else:
                # Flattened schedule: reshape into weeks, then take first 52
                schedule_array = np.array(schedule)
                num_weeks = len(schedule_array) // week_len
                # Reshape: (num_weeks, week_len), then take first 52
                weeks_reshaped = schedule_array[:num_weeks * week_len].reshape(num_weeks, week_len)
                W = weeks_reshaped[:52]  # shape (52, week_len) or less if not enough weeks
            avg_week = W.mean(axis=0) 
            
       

        # ---- plot ----
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week[0], label=label1, linewidth=1.5)
        ax.plot(x, avg_week[1], label=label2, linewidth=1.5)

        # Major ticks: day labels at center of each day
        entries_per_day = timesteps_per_day
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
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

        ax.set_ylim(-0.1, 1.1)

        # Handle path
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


    

class OccupantBehavior(BaseModel):
    """ Represents the behavior profile of a household occupant, including occupancy, lighting, equipment, and HVAC. """
    model_config = ConfigDict(validate_assignment=True)
    num_occupants: int = Field(..., description="Number of occupants in the household.")
    occupancy_schedule: list[float] = Field(..., description="Occupancy schedule as a list of floats as a fraction of occupancy.")
    lighting_schedule: list[float] = Field(..., description="Lighting schedule as a list of fractions representing lighting usage.")
    lighting_if_dimming: float = Field(..., description="Indicates the probability of dimming being used.")
    equipment_schedule: list[float] = Field(..., description="Equipment schedule as a list of floats representing equipment usage for the household (W).")
    dhw_max_flow_rate_m3_per_s: float = Field(..., description="Maximum flow rate of domestic hot water in cubic meters per second.")
    dhw_schedule: list[float] = Field(..., description="Domestic hot water usage schedule as a list of fractions (relative to the max flow rate) representing flow rate.")    

    @staticmethod
    def aggregate_occupant_behavior(occupant_behaviors: list["OccupantBehavior"]) -> "OccupantBehavior":
        """ Aggregates occupant behaviors into a single occupant behavior. """
        num_occupants = sum(behavior.num_occupants for behavior in occupant_behaviors)
        occupancy_schedule = [0.0] * len(occupant_behaviors[0].occupancy_schedule)
        lighting_schedule = [0.0] * len(occupant_behaviors[0].lighting_schedule)
        equipment_schedule = [0.0] * len(occupant_behaviors[0].equipment_schedule)
        dhw_max_flow_rate_m3_per_schedule = [0.0] * len(occupant_behaviors[0].dhw_schedule)
        lighting_dimming_probability = 0.0

        for behavior in occupant_behaviors:
            lighting_dimming_probability += behavior.lighting_if_dimming * behavior.num_occupants / num_occupants
            for i in range(len(behavior.occupancy_schedule)):
                occupancy_schedule[i] += behavior.occupancy_schedule[i] * behavior.num_occupants / num_occupants
                equipment_schedule[i] += behavior.equipment_schedule[i] * behavior.num_occupants
                dhw_max_flow_rate_m3_per_schedule[i] += behavior.dhw_max_flow_rate_m3_per_s * behavior.dhw_schedule[i]
                lighting_schedule[i] += behavior.lighting_schedule[i] * behavior.num_occupants  / num_occupants

        dhw_max_flow_rate_m3_per_s = max(dhw_max_flow_rate_m3_per_schedule)
        dhw_schedule = [dhw / dhw_max_flow_rate_m3_per_s for dhw in dhw_max_flow_rate_m3_per_schedule]
            
        return OccupantBehavior(
            num_occupants=sum(behavior.num_occupants for behavior in occupant_behaviors),
            occupancy_schedule=occupancy_schedule,
            lighting_schedule=lighting_schedule,
            lighting_if_dimming=lighting_dimming_probability,
            equipment_schedule=equipment_schedule,
            dhw_max_flow_rate_m3_per_s=dhw_max_flow_rate_m3_per_s,
            dhw_schedule=dhw_schedule,
        )

    
    def to_OB_annual(self, resolution_mins, occupant_profile: Occupant) -> "OccupantBehavior":
        """ Generate annual occupancy behavior schedules. """
        occ_gen = OccupancyGenerator( occupant_profile, OccupancyAssumptions.default(), resolution_mins)
        occupancy_schedule,sleep_schedule = occ_gen.household_annual_schedule()
        occupancy_mask, active_mask, sleep_mask = occ_gen.get_annual_mask(occupancy_schedule, sleep_schedule)

        lighting_gen = LightingGenerator(self.lighting)
        if_dimming = lighting_gen.get_dimming()
        lighting_schedule = lighting_gen.lighting_annual_schedule(occupancy_mask, sleep_mask)

        equipment_gen = EquipmentGenerator(self.equipment,active_mask,sleep_mask, occupancy_schedule, self.occupancy.num_occupants, resolution_mins, EquipmentAssumptions.default())
        equipment_schedule, laundry_cycles, dishwasher_cycles = equipment_gen.equipment_annual_schedule()

        dhw_gen = DHWGenerator( DHWAssumptions.default(), self.equipment, self.occupancy.num_occupants, laundry_cycles, dishwasher_cycles, resolution_mins)
        flow_rate, dhw_schedule = dhw_gen.dhw_annual_schedule()
        return OccupantBehavior(
            num_occupants=self.occupancy.num_occupants,
            occupancy_schedule=ScheduleUtils.flatten_schedule(occupancy_schedule),
            lighting_schedule=ScheduleUtils.flatten_schedule(lighting_schedule),
            lighting_if_dimming= 1.0 if if_dimming else 0.0,
            equipment_schedule=ScheduleUtils.flatten_schedule(equipment_schedule),
            dhw_max_flow_rate_m3_per_s=flow_rate,
            dhw_schedule=ScheduleUtils.flatten_schedule(dhw_schedule)
        )
    
    
    @staticmethod
    def multiple_to_OB_annual(occupants: dict["Occupant", int], resolution_mins: int) -> "OccupantBehavior":
        """ Generate annual aggregated occupancy behavior schedules for multiple occupants. """
        behaviors = []
        for occupant, count in occupants.items():
            behavior = occupant.to_OB_annual(resolution_mins, occupant)
            for _ in range(count):
                behaviors.append(behavior)
        
        return OccupantBehavior.aggregate_occupant_behavior(behaviors)
        




