"""
Unit tests for occupancy generator.

Tests the occupancy generation logic in src/generator/occupancy_generator.py
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from pathlib import Path

from generator.occupancy_generator import (
    OccupantRole,
    RoleAssumption,
    OccupancyAssumptions,
    TimeRangeDistribution,
    SingleOccupantTracker,
    OccupancyGenerator
)
from model.occupancy import (
    Occupancy,
    HouseholdComposition,
    TimeRange,
    WeekdayOccupancyPattern,
    WeekendOccupancyPattern
)


class TestTimeRangeDistribution:
    """Tests for TimeRangeDistribution."""

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

    



class TestOccupancyGenerator:
    """Tests for OccupancyGenerator."""

    @pytest.mark.parametrize("start, end", [
        (22.0, 6.0),
        (1.0, 9.0),
        (0.5, 8.5)
    ])
    def test_get_sleep_mask(self, start, end):
        """Test sleep mask generation."""
        sleep_schedule = [TimeRange(start_hour=22.0, end_hour=6.0)] * 7
        num_per_hour = 4  # 15-minute resolution

        mask = OccupancyGenerator.get_sleep_mask(sleep_schedule, num_per_hour)

        assert len(mask) == 7 * 24 * 4
        assert any(mask)
        assert not all(mask)
        assert sum(mask) == 7 * 8 * 4  


    def test_always_occupied_schedule(self):
        """Test schedule when always occupied on weekdays."""
        occupancy = Occupancy(
            num_occupants=1,
            household_composition=HouseholdComposition(stayathome=1),
            weekday_pattern=WeekdayOccupancyPattern(is_always_occupied=True),
            weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True),
            sleep_time=TimeRange(start_hour=22.0, end_hour=6.0)
        )
        assumptions = OccupancyAssumptions.default()
        generator = OccupancyGenerator(occupancy, assumptions, resolution_mins=60)
        sleep = generator.household_sleep_schedule()
        occupancy = generator.household_fullweek_schedule(sleep)
        # All hours should be occupied (1.0)
        assert len(occupancy) == 7 * 24
        assert all(o == 1.0 for o in occupancy)
        

    @pytest.mark.parametrize("res_mins", [5,10,15,30,60])
    def test_week_schedule(self, res_mins):
        """Test that weekday schedule has correct length for different resolutions."""
        occupancy = Occupancy(
            num_occupants=2,
            household_composition=HouseholdComposition(daily_commuter=2),
            weekday_pattern=WeekdayOccupancyPattern(
                is_always_occupied=False,
                away_interval=TimeRange(start_hour=9.0, end_hour=17.0),
                num_of_days=5
            ),
            weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True),
            sleep_time=TimeRange(start_hour=22.0, end_hour=6.0)
        )
        assumptions = OccupancyAssumptions.default()

        gen = OccupancyGenerator(occupancy, assumptions, resolution_mins=res_mins)
        sleep = gen.household_sleep_schedule()
        sleep_mask = OccupancyGenerator.get_sleep_mask(sleep, gen.num_per_hour)
        schedule = gen.household_fullweek_schedule(sleep)

        assert len(schedule) == 7 * 24 * (60 // res_mins)
        # Check that during sleep hours, occupancy is always 1
        for i in range(len(schedule)):
            if sleep_mask[i]:
                assert schedule[i] == 1.0

    
    @pytest.mark.parametrize("res_mins", [5,10,15,30,60])
    def test_annual_schedule_length(self, res_mins):
        """Test that annual schedule has correct length for different resolutions."""
        occupancy = Occupancy(
            num_occupants=2,
            household_composition=HouseholdComposition(daily_commuter=2),
            weekday_pattern=WeekdayOccupancyPattern(
                is_always_occupied=False,
                away_interval=TimeRange(start_hour=11.0, end_hour=16.0),
                num_of_days=5
            ),
            weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True),
            sleep_time=TimeRange(start_hour=22.0, end_hour=6.0)
        )
        assumptions = OccupancyAssumptions.default()

        gen = OccupancyGenerator(occupancy, assumptions, resolution_mins=res_mins)
        annual_schedule, annual_sleep = gen.household_annual_schedule()

        assert len(annual_schedule) == 53  # 52 full weeks + 1 day
        for i in range(52):
            assert len(annual_schedule[i]) == 7 * 24 * (60 // res_mins)
            assert len(annual_sleep[i]) == 7 * 24 * (60 // res_mins)
            for j in range(len(annual_schedule[i])):
                if annual_sleep[i][j]:
                    assert annual_schedule[i][j] == 1.0
        assert len(annual_schedule[52]) == 24 * (60 // res_mins)
        assert len(annual_sleep[52]) == 24 * (60 // res_mins)
        for j in range(len(annual_schedule[52])):
            if annual_sleep[52][j]:
                assert annual_schedule[52][j] == 1.0



    @pytest.mark.parametrize("start,end", [
        (9.0, 17.0),
        (8.5, 16.5),
        (10.0, 15.0)
    ])
    def test_weekday_away_interval(self,start,end):
        """Test that weekday away interval is respected in schedule."""
        occupancy = Occupancy(
            num_occupants=1,
            household_composition=HouseholdComposition(daily_commuter=1),
            weekday_pattern=WeekdayOccupancyPattern(
                is_always_occupied=False,
                away_interval=TimeRange(start_hour=start, end_hour=end),
                num_of_days=5
            ),
            weekend_pattern=WeekendOccupancyPattern(is_always_occupied=True),
            sleep_time=TimeRange(start_hour=22.0, end_hour=6.0)
        )
        assumptions = OccupancyAssumptions.default()
        generator = OccupancyGenerator(occupancy, assumptions, resolution_mins=30)

        
        starts = []
        ends = []
        for _ in range(20):
            sleep = generator.household_sleep_schedule()
            weekday_away,_ = generator.weekday_away_interval(sleep[:5])
            for wa in weekday_away:
                starts.append(wa.start_hour)
                ends.append(wa.end_hour)
                
        
        start_mean = np.mean(starts)
        end_mean = np.mean(ends)
        assert abs(start_mean - start) / abs(start) < 0.05
        assert abs(end_mean - end) / abs(end) < 0.05


    @pytest.mark.statistical
    def test_3people_family_schedule(self):
        """Test weekday schedule for a family with mixed roles."""

        occupancy_json = """
        {
            "num_occupants": 3,
            "household_composition": {
            "daily_commuter": 2,
            "hybrid_worker": 0,
            "stayathome": 0,
            "k12_or_daycare": 1,
            "college_student": 0
            },
            "weekday_pattern": {
            "is_always_occupied": false,
            "away_interval": {
                "start_hour": 9,
                "end_hour": 17
            },
            "num_of_days": 5
            },
            "weekend_pattern": {
            "is_always_occupied": true
            },
            "sleep_time": {
            "start_hour": 22,
            "end_hour": 6
            }
        }
        """
        occupancy = Occupancy.model_validate_json(occupancy_json)

        assumptions = OccupancyAssumptions.default()
        generator = OccupancyGenerator(occupancy, assumptions, resolution_mins=15)

        annual_schedule, annual_sleep = generator.household_annual_schedule()

         # ---- average ONLY the first 52 weeks (exclude the last extra day/week entry) ----
        weeks = annual_schedule[:52]  # each is a list[float] of length 7*24*num_per_hour
        sleep_weeks = annual_sleep[:52]  # each is a list[bool] of length 7*24*num_per_hour
        week_len = 7 * 24 * generator.num_per_hour

        # sanity
        assert len(weeks) == 52
        assert all(len(w) == week_len for w in weeks)
        assert len(sleep_weeks) == 52
        assert all(len(s) == week_len for s in sleep_weeks)

        # Average occupancy schedule
        W = np.array(weeks, dtype=float)          # shape (52, week_len)
        avg_week = W.mean(axis=0)                 # shape (week_len,)

        # Average sleep schedule: convert True (sleep) to 1, False (awake) to 0
        S = np.array([[1.0 if sleeping else 0.0 for sleeping in sleep_week]
                      for sleep_week in sleep_weeks], dtype=float)  # shape (52, week_len)
        avg_sleep = S.mean(axis=0)                # shape (week_len,)

        # ---- plot 1 ----
        fig = plt.figure(figsize=(12,3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week, label='Occupancy', linewidth=1.5)
        ax.plot(x, avg_sleep, label='Sleep', linestyle='--', linewidth=1.5, alpha=0.7)

        # Major ticks: day labels at center of each day
        entries_per_day = 24 * generator.num_per_hour
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlim(0, week_len)

        # Minor ticks: hour markers at 0, 6, 18 for each day with labels
        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + hour * generator.num_per_hour)
                hour_labels.append(str(hour))

        ax2 = ax.twiny()  # Create secondary x-axis for hour labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        

        # Add vertical grid lines at hour markers
        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)

        # Add dotted lines for work hours (9 AM and 5 PM) for each day
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [9, 17]:
                work_hour_tick = day_start + hour * generator.num_per_hour
                ax.axvline(x=work_hour_tick, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)

        ax.set_title("Averaged Weekly Occupancy and Sleep")
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')

        ax.set_ylim(-0.1, 1.1)

        unit_dir = Path(__file__).resolve().parent
        out_dir = unit_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "avg_week_occupancy_3pplfamily.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # ---- plot 2: Heatmap ----
        # Flatten all 365 days into continuous array
        hours_per_day = 24 * generator.num_per_hour

        # Concatenate all weeks (52 full weeks + 1 extra day)
        all_days = []
        for week_idx, week_data in enumerate(annual_schedule):
            if week_idx < 52:
                # Full week: reshape into 7 days
                week_array = np.array(week_data)
                for day in range(7):
                    day_data = week_array[day * hours_per_day:(day + 1) * hours_per_day]
                    all_days.append(day_data)
            else:
                # Last partial week (1 day)
                all_days.append(np.array(week_data))

        # Stack into 2D array: rows = time slots, cols = days
        heatmap_data = np.column_stack(all_days)  # Shape: (hours_per_day, 365)

        fig2, ax_heat = plt.subplots(figsize=(12, 3))

        # Create heatmap
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlGn', vmin=0, vmax=1, origin='upper')

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax_heat)
        cbar.set_label('Occupancy Fraction', rotation=270, labelpad=20)

        # Set x-axis (days)
        # Show ticks at week boundaries (every 7 days)
        week_ticks = [w * 7 for w in range(0, 53, 4)]  # Every 4 weeks
        ax_heat.set_xticks(week_ticks)
        ax_heat.set_xticklabels([f'Week {w}' for w in range(0, 53, 4)])
        ax_heat.set_xlabel('Day of Year')

        # Set y-axis (hours of day)
        # Show hour labels at every 2 hours
        hour_tick_positions = [h * generator.num_per_hour for h in range(0, 25, 2)]
        ax_heat.set_yticks(hour_tick_positions)
        ax_heat.set_yticklabels([str(h) for h in range(0, 25, 2)])
        ax_heat.set_ylabel('Hour of Day')

        # Add horizontal lines for work hours (9 AM and 5 PM)
        for hour in [9, 17]:
            hour_position = hour * generator.num_per_hour
            ax_heat.axhline(y=hour_position, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)

        ax_heat.set_title('Annual Occupancy Heatmap (365 Days)')

        fig2.tight_layout()
        out_path_heat = out_dir / "annual_occupancy_heatmap_3pplfamily.png"
        fig2.savefig(out_path_heat, dpi=200)
        plt.close(fig2)


    @pytest.mark.statistical
    def test_4students_schedule(self):
        """Test weekday schedule for a family with mixed roles."""
        occupancy_json = """
        {
            "num_occupants": 4,
            "household_composition": {
                "daily_commuter": 0,
                "hybrid_worker": 0,
                "stayathome": 0,
                "k12_or_daycare": 0,
                "college_student": 4
            },
            "weekday_pattern": {
                "is_always_occupied": false,
                "away_interval": {
                    "start_hour": 11,
                    "end_hour": 15
                },
                "num_of_days": 4
            },
            "weekend_pattern": {
                "is_always_occupied": false,
                "away_interval": {
                    "start_hour": 14,
                    "end_hour": 16
                }
            },
            "sleep_time": {
                "start_hour": 0,
                "end_hour": 8
            }
        }
        """
        occupancy = Occupancy.model_validate_json(occupancy_json)
        assumptions = OccupancyAssumptions.default()
        generator = OccupancyGenerator(occupancy, assumptions, resolution_mins=15)

        annual_schedule, annual_sleep = generator.household_annual_schedule()

         # ---- average ONLY the first 52 weeks (exclude the last extra day/week entry) ----
        weeks = annual_schedule[:52]  # each is a list[float] of length 7*24*num_per_hour
        sleep_weeks = annual_sleep[:52]  # each is a list[bool] of length 7*24*num_per_hour
        week_len = 7 * 24 * generator.num_per_hour

        # sanity
        assert len(weeks) == 52
        assert all(len(w) == week_len for w in weeks)
        assert len(sleep_weeks) == 52
        assert all(len(s) == week_len for s in sleep_weeks)

        # Average occupancy schedule
        W = np.array(weeks, dtype=float)          # shape (52, week_len)
        avg_week = W.mean(axis=0)                 # shape (week_len,)

        # Average sleep schedule: convert True (sleep) to 1, False (awake) to 0
        S = np.array([[1.0 if sleeping else 0.0 for sleeping in sleep_week]
                      for sleep_week in sleep_weeks], dtype=float)  # shape (52, week_len)
        avg_sleep = S.mean(axis=0)                # shape (week_len,)

        # ---- plot 1 ----
        fig = plt.figure(figsize=(12,3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week, label='Occupancy', linewidth=1.5)
        ax.plot(x, avg_sleep, label='Sleep', linestyle='--', linewidth=1.5, alpha=0.7)

        # Major ticks: day labels at center of each day
        entries_per_day = 24 * generator.num_per_hour
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlim(0, week_len)

        # Minor ticks: hour markers at 0, 6, 18 for each day with labels
        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + hour * generator.num_per_hour)
                hour_labels.append(str(hour))

        ax2 = ax.twiny()  # Create secondary x-axis for hour labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        

        # Add vertical grid lines at hour markers
        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)

        # # Add dotted lines for work hours (9 AM and 5 PM) for each day
        for day in range(5):
            day_start = day * entries_per_day
            for hour in [11, 15]:
                work_hour_tick = day_start + hour * generator.num_per_hour
                ax.axvline(x=work_hour_tick, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)
        for day in range(5,7):
            day_start = day * entries_per_day
            for hour in [14, 16]:
                work_hour_tick = day_start + hour * generator.num_per_hour
                ax.axvline(x=work_hour_tick, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)

        ax.set_title("Averaged Weekly Occupancy and Sleep")
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')

        ax.set_ylim(-0.1, 1.1)

        unit_dir = Path(__file__).resolve().parent
        out_dir = unit_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "avg_week_occupancy_4students.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # ---- plot 2: Heatmap ----
        # Flatten all 365 days into continuous array
        hours_per_day = 24 * generator.num_per_hour

        # Concatenate all weeks (52 full weeks + 1 extra day)
        all_days = []
        for week_idx, week_data in enumerate(annual_schedule):
            if week_idx < 52:
                # Full week: reshape into 7 days
                week_array = np.array(week_data)
                for day in range(7):
                    day_data = week_array[day * hours_per_day:(day + 1) * hours_per_day]
                    all_days.append(day_data)
            else:
                # Last partial week (1 day)
                all_days.append(np.array(week_data))

        # Stack into 2D array: rows = time slots, cols = days
        heatmap_data = np.column_stack(all_days)  # Shape: (hours_per_day, 365)

        fig2, ax_heat = plt.subplots(figsize=(12, 3))

        # Create heatmap
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlGn', vmin=0, vmax=1, origin='upper')

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax_heat)
        cbar.set_label('Occupancy Fraction', rotation=270, labelpad=20)

        # Set x-axis (days)
        # Show ticks at week boundaries (every 7 days)
        week_ticks = [w * 7 for w in range(0, 53, 4)]  # Every 4 weeks
        ax_heat.set_xticks(week_ticks)
        ax_heat.set_xticklabels([f'Week {w}' for w in range(0, 53, 4)])
        ax_heat.set_xlabel('Day of Year')

        # Set y-axis (hours of day)
        # Show hour labels at every 2 hours
        hour_tick_positions = [h * generator.num_per_hour for h in range(0, 25, 2)]
        ax_heat.set_yticks(hour_tick_positions)
        ax_heat.set_yticklabels([str(h) for h in range(0, 25, 2)])
        ax_heat.set_ylabel('Hour of Day')

        # Add horizontal lines for work hours (9 AM and 5 PM)
        for hour in [11, 15]:
            hour_position = hour * generator.num_per_hour
            ax_heat.axhline(y=hour_position, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)

        ax_heat.set_title('Annual Occupancy Heatmap (365 Days)')

        fig2.tight_layout()
        out_path_heat = out_dir / "annual_occupancy_heatmap_4students.png"
        fig2.savefig(out_path_heat, dpi=200)
        plt.close(fig2)


    @pytest.mark.statistical
    def test_2hybrid_schedule(self):
        """Test weekday schedule for a family with mixed roles."""
        occupancy_json = """
        {
            "num_occupants": 2,
            "household_composition": {
                "daily_commuter": 0,
                "hybrid_worker": 2,
                "stayathome": 0,
                "k12_or_daycare": 0,
                "college_student": 0
            },
            "weekday_pattern": {
                "is_always_occupied": false,
                "away_interval": {
                    "start_hour": 8,
                    "end_hour": 17
                },
                "num_of_days": 3
            },
            "weekend_pattern": {
                "is_always_occupied": false,
                "away_interval": {
                    "start_hour": 16,
                    "end_hour": 22
                }
            },
            "sleep_time": {
                "start_hour": 0,
                "end_hour": 8
            }
        }
        """
        occupancy = Occupancy.model_validate_json(occupancy_json)
        assumptions = OccupancyAssumptions.default()
        generator = OccupancyGenerator(occupancy, assumptions, resolution_mins=15)

        annual_schedule, annual_sleep = generator.household_annual_schedule()

         # ---- average ONLY the first 52 weeks (exclude the last extra day/week entry) ----
        weeks = annual_schedule[:52]  # each is a list[float] of length 7*24*num_per_hour
        sleep_weeks = annual_sleep[:52]  # each is a list[bool] of length 7*24*num_per_hour
        week_len = 7 * 24 * generator.num_per_hour

        # sanity
        assert len(weeks) == 52
        assert all(len(w) == week_len for w in weeks)
        assert len(sleep_weeks) == 52
        assert all(len(s) == week_len for s in sleep_weeks)

        # Average occupancy schedule
        W = np.array(weeks, dtype=float)          # shape (52, week_len)
        avg_week = W.mean(axis=0)                 # shape (week_len,)

        # Average sleep schedule: convert True (sleep) to 1, False (awake) to 0
        S = np.array([[1.0 if sleeping else 0.0 for sleeping in sleep_week]
                      for sleep_week in sleep_weeks], dtype=float)  # shape (52, week_len)
        avg_sleep = S.mean(axis=0)                # shape (week_len,)

        # ---- plot 1 ----
        fig = plt.figure(figsize=(12,3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week, label='Occupancy', linewidth=1.5)
        ax.plot(x, avg_sleep, label='Sleep', linestyle='--', linewidth=1.5, alpha=0.7)

        # Major ticks: day labels at center of each day
        entries_per_day = 24 * generator.num_per_hour
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlim(0, week_len)

        # Minor ticks: hour markers at 0, 6, 18 for each day with labels
        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + hour * generator.num_per_hour)
                hour_labels.append(str(hour))

        ax2 = ax.twiny()  # Create secondary x-axis for hour labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        

        # Add vertical grid lines at hour markers
        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)

        # Add dotted lines for work hours (9 AM and 5 PM) for each day
        for day in range(5):
            day_start = day * entries_per_day
            for hour in [8, 17]:
                work_hour_tick = day_start + hour * generator.num_per_hour
                ax.axvline(x=work_hour_tick, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)
        for day in range(5,7):
            day_start = day * entries_per_day
            for hour in [16, 22]:
                work_hour_tick = day_start + hour * generator.num_per_hour
                ax.axvline(x=work_hour_tick, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)

        ax.set_title("Averaged Weekly Occupancy and Sleep")
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')

        ax.set_ylim(-0.1, 1.1)

        unit_dir = Path(__file__).resolve().parent
        out_dir = unit_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "avg_week_occupancy_2hybrid.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # ---- plot 2: Heatmap ----
        # Flatten all 365 days into continuous array
        hours_per_day = 24 * generator.num_per_hour

        # Concatenate all weeks (52 full weeks + 1 extra day)
        all_days = []
        for week_idx, week_data in enumerate(annual_schedule):
            if week_idx < 52:
                # Full week: reshape into 7 days
                week_array = np.array(week_data)
                for day in range(7):
                    day_data = week_array[day * hours_per_day:(day + 1) * hours_per_day]
                    all_days.append(day_data)
            else:
                # Last partial week (1 day)
                all_days.append(np.array(week_data))

        # Stack into 2D array: rows = time slots, cols = days
        heatmap_data = np.column_stack(all_days)  # Shape: (hours_per_day, 365)

        fig2, ax_heat = plt.subplots(figsize=(12, 3))

        # Create heatmap
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlGn', vmin=0, vmax=1, origin='upper')

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax_heat)
        cbar.set_label('Occupancy Fraction', rotation=270, labelpad=20)

        # Set x-axis (days)
        # Show ticks at week boundaries (every 7 days)
        week_ticks = [w * 7 for w in range(0, 53, 4)]  # Every 4 weeks
        ax_heat.set_xticks(week_ticks)
        ax_heat.set_xticklabels([f'Week {w}' for w in range(0, 53, 4)])
        ax_heat.set_xlabel('Day of Year')

        # Set y-axis (hours of day)
        # Show hour labels at every 2 hours
        hour_tick_positions = [h * generator.num_per_hour for h in range(0, 25, 2)]
        ax_heat.set_yticks(hour_tick_positions)
        ax_heat.set_yticklabels([str(h) for h in range(0, 25, 2)])
        ax_heat.set_ylabel('Hour of Day')

        # Add horizontal lines for work hours (9 AM and 5 PM)
        for hour in [8, 17]:
            hour_position = hour * generator.num_per_hour
            ax_heat.axhline(y=hour_position, color='purple', alpha=0.5, linestyle=':', linewidth=1.0)

        ax_heat.set_title('Annual Occupancy Heatmap (365 Days)')

        fig2.tight_layout()
        out_path_heat = out_dir / "annual_occupancy_heatmap_2hybrid.png"
        fig2.savefig(out_path_heat, dpi=200)
        plt.close(fig2)

    @pytest.mark.statistical
    def test_largefamily_schedule(self):
        """Test weekday schedule for a family with mixed roles."""
        occupancy_json = """
        {
            "num_occupants": 7,
            "household_composition": {
                "daily_commuter": 1,
                "hybrid_worker": 1,
                "stayathome": 2,
                "k12_or_daycare": 2,
                "college_student": 1
            },
            "weekday_pattern": {
                "is_always_occupied": true
            },
            "weekend_pattern": {
                "is_always_occupied": true
            },
            "sleep_time": {
                "start_hour": 2,
                "end_hour": 6
            }
        }
        """
        occupancy = Occupancy.model_validate_json(occupancy_json)
        assumptions = OccupancyAssumptions.default()
        generator = OccupancyGenerator(occupancy, assumptions, resolution_mins=15)

        annual_schedule, annual_sleep = generator.household_annual_schedule()

         # ---- average ONLY the first 52 weeks (exclude the last extra day/week entry) ----
        weeks = annual_schedule[:52]  # each is a list[float] of length 7*24*num_per_hour
        sleep_weeks = annual_sleep[:52]  # each is a list[bool] of length 7*24*num_per_hour
        week_len = 7 * 24 * generator.num_per_hour

        # sanity
        assert len(weeks) == 52
        assert all(len(w) == week_len for w in weeks)
        assert len(sleep_weeks) == 52
        assert all(len(s) == week_len for s in sleep_weeks)

        # Average occupancy schedule
        W = np.array(weeks, dtype=float)          # shape (52, week_len)
        avg_week = W.mean(axis=0)                 # shape (week_len,)

        # Average sleep schedule: convert True (sleep) to 1, False (awake) to 0
        S = np.array([[1.0 if sleeping else 0.0 for sleeping in sleep_week]
                      for sleep_week in sleep_weeks], dtype=float)  # shape (52, week_len)
        avg_sleep = S.mean(axis=0)                # shape (week_len,)

        # ---- plot 1 ----
        fig = plt.figure(figsize=(12,3))
        ax = fig.add_subplot(111)
        x = np.arange(week_len)
        ax.plot(x, avg_week, label='Occupancy', linewidth=1.5)
        ax.plot(x, avg_sleep, label='Sleep', linestyle='--', linewidth=1.5, alpha=0.7)

        # Major ticks: day labels at center of each day
        entries_per_day = 24 * generator.num_per_hour
        day_centers = [(d * entries_per_day + entries_per_day / 2) for d in range(7)]
        ax.set_xticks(day_centers)
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlim(0, week_len)

        # Minor ticks: hour markers at 0, 6, 18 for each day with labels
        hour_ticks = []
        hour_labels = []
        for day in range(7):
            day_start = day * entries_per_day
            for hour in [0, 6, 12, 18]:
                hour_ticks.append(day_start + hour * generator.num_per_hour)
                hour_labels.append(str(hour))

        ax2 = ax.twiny()  # Create secondary x-axis for hour labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, fontsize=8, color='gray')

        

        # Add vertical grid lines at hour markers
        for tick in hour_ticks:
            ax.axvline(x=tick, color='gray', alpha=0.2, linestyle=':', linewidth=0.5)

      
        ax.set_title("Averaged Weekly Occupancy and Sleep")
        ax.set_xlabel("Time in a Week")
        ax.set_ylabel("Fraction")
        ax.legend(loc='upper right')

        ax.set_ylim(-0.1, 1.1)

        unit_dir = Path(__file__).resolve().parent
        out_dir = unit_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "avg_week_occupancy_largefamily.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # ---- plot 2: Heatmap ----
        # Flatten all 365 days into continuous array
        hours_per_day = 24 * generator.num_per_hour

        # Concatenate all weeks (52 full weeks + 1 extra day)
        all_days = []
        for week_idx, week_data in enumerate(annual_schedule):
            if week_idx < 52:
                # Full week: reshape into 7 days
                week_array = np.array(week_data)
                for day in range(7):
                    day_data = week_array[day * hours_per_day:(day + 1) * hours_per_day]
                    all_days.append(day_data)
            else:
                # Last partial week (1 day)
                all_days.append(np.array(week_data))

        # Stack into 2D array: rows = time slots, cols = days
        heatmap_data = np.column_stack(all_days)  # Shape: (hours_per_day, 365)

        fig2, ax_heat = plt.subplots(figsize=(12, 3))

        # Create heatmap
        im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlGn', vmin=0, vmax=1, origin='upper')

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax_heat)
        cbar.set_label('Occupancy Fraction', rotation=270, labelpad=20)

        # Set x-axis (days)
        # Show ticks at week boundaries (every 7 days)
        week_ticks = [w * 7 for w in range(0, 53, 4)]  # Every 4 weeks
        ax_heat.set_xticks(week_ticks)
        ax_heat.set_xticklabels([f'Week {w}' for w in range(0, 53, 4)])
        ax_heat.set_xlabel('Day of Year')

        # Set y-axis (hours of day)
        # Show hour labels at every 2 hours
        hour_tick_positions = [h * generator.num_per_hour for h in range(0, 25, 2)]
        ax_heat.set_yticks(hour_tick_positions)
        ax_heat.set_yticklabels([str(h) for h in range(0, 25, 2)])
        ax_heat.set_ylabel('Hour of Day')

     

        ax_heat.set_title('Annual Occupancy Heatmap (365 Days)')

        fig2.tight_layout()
        out_path_heat = out_dir / "annual_occupancy_heatmap_largefamily.png"
        fig2.savefig(out_path_heat, dpi=200)
        plt.close(fig2)

    def test_revise_by_sleep(self):
        """Test revision of schedule based on sleep mask."""
        sleep_mask = [True, False, True, False, True]
        schedule = [1.0, 1.0, 1.0, 1.0, 1.0]
        value = 0.5

        revised = OccupancyGenerator.revise_by_sleep(sleep_mask, schedule, value)

        assert revised == [0.5, 1.0, 0.5, 1.0, 0.5]

    def test_get_occupancy_mask(self):
        """Test occupancy mask generation."""
        schedule = [0.0, 0.5, 1.0, 0.0, 0.75]

        mask = OccupancyGenerator.get_occupancy_mask(schedule)

        assert mask == [False, True, True, False, True]

    def test_revise_by_absence(self):
        """Test revision of schedule based on occupancy mask."""
        occupancy_mask = [True, False, True, False, True]
        schedule = [10, 20, 30, 40, 50]
        value = 0

        revised = OccupancyGenerator.revise_by_absence(occupancy_mask, schedule, value)

        assert revised == [10, 0, 30, 0, 50]
