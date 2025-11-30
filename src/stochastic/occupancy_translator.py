from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import stochastic.translation_rule as TranslationRule

class OccupancyTranslator:

    def __init__(self, occupancy: Occupancy.Occupancy):
        self.occupancy = occupancy
        self.weekly_schedule: list[float] = []
        self.sleep_schedule: list[tuple[int,int] | None] = None

    @staticmethod
    def _sample_single_occupant( start_dist, end_dist, wraps_midnight: bool = False) -> tuple[int, int]:
        """
        Samples a single occupant's leave and return times ensuring logical consistency.
        
        Args:
            start_dist: Distribution for leave time
            end_dist: Distribution for return time
            wraps_midnight: If True, allows return time < leave time (crosses midnight)
        
        Returns:
            Tuple of (start_time, end_time)
        """
        start_time = start_dist.sample()
        while True:
            end_time = end_dist.sample()
            if wraps_midnight or end_time > start_time:
                return start_time, end_time    
            
    @staticmethod
    def get_schedule_by_no_one_home(
        leaves: list[list[int]], 
        returns: list[list[int]], 
        num_occupants: int,
        no_one_home_ranges: list[Occupancy.TimeRange]
    ) -> list[float]:
        """
        Generates the final schedule with strict enforcement of the 'Last Man Standing'
        and 'Absolute Zero' region through no one home period from occupant profile.
        """
        schedule = []
        ratio = 1.0 / num_occupants
        

        for day_idx in range(len(leaves)):
            day_leaves = leaves[day_idx]
            day_returns = returns[day_idx]
            no_one_home_range = no_one_home_ranges[day_idx]

            for h in range(24):
                # --- REGION 1: ABSOLUTE ZERO (Hard Constraint) ---
                if no_one_home_range.contains_hour(h):
                    schedule.append(0.0)
                    continue 
                
                # --- REGION 2: CALCULATE RAW PRESENCE ---
                people_present = 0
                for i in range(num_occupants):
                    leave_time = day_leaves[i]
                    return_time = day_returns[i]
                    away_range = Occupancy.TimeRange(start_hour=leave_time, end_hour=return_time)
                    if not away_range.contains_hour(h):
                       people_present += 1

                # --- REGION 3: LAST MAN STANDING (The Fix) ---
                # If no one is calculated to be present but we're outside the 
                # absolute zero range, ensure at least one person is home
                if people_present == 0:
                    people_present = 1
                
                # Calculate fraction
                frac = round(people_present * ratio, 4)
                schedule.append(frac)

        return schedule
    
    @staticmethod
    def _get_schedule_no_constraint(
        leaves: list[list[int]], 
        returns: list[list[int]], 
        num_occupants: int,
        ) -> list[float]:
        """basic schedule generator without hard constraints"""
        schedule = []
        ratio = 1.0 / num_occupants
        for day_idx in range(len(leaves)):
            day_leaves = leaves[day_idx]
            day_returns = returns[day_idx]
            for h in range(24):
                people_present = 0
                for i in range(num_occupants):
                    leave_time = day_leaves[i]
                    return_time = day_returns[i]
                    away_range = Occupancy.TimeRange(start_hour=leave_time, end_hour=return_time)
                    if not away_range.contains_hour(h):
                       people_present += 1
                frac = round(people_present * ratio, 4)
                schedule.append(frac)
        return schedule

    @staticmethod
    def _is_time_range_occupied(self, day_offset: int, start_hour: int, end_hour: int, wraps_midnight: bool = False) -> bool:
        """
        Checks if all hours in a time range are occupied (non-zero) in the weekly schedule.
        Args:
            day_offset: Day index (0-6) within the week
            start_hour: Starting hour (0-23)
            end_hour: Ending hour (0-23)
            wraps_midnight: If True, the range crosses midnight (e.g., 23:00 to 6:00)

        Returns:
            True if all hours in the range are occupied, False otherwise
        """
        if wraps_midnight:
            # Check from start_hour to end of day
            for hour in range(start_hour, 24):
                if self.weekly_schedule[day_offset * 24 + hour] == 0.0:
                    return False
            # Check from start of next day to end_hour
            next_day_offset = (day_offset + 1) % 7
            for hour in range(0, end_hour):
                if self.weekly_schedule[next_day_offset * 24 + hour] == 0.0:
                    return False
        else:
            # Normal range within same day
            for hour in range(start_hour, end_hour):
                if self.weekly_schedule[day_offset * 24 + hour] == 0.0:
                    return False
        return True
    
    def translate_occupancy_presumption_weekday(self) -> list[float]:
        occupancy = self.occupancy
        # Determine if there's a no-one-home period
        if occupancy.weekday_no_one_home is None:
            # No one leaves - everyone stays home all day
            no_one_home_range = Occupancy.TimeRange(start_hour=23, end_hour=0)  # Zero duration
        else:
            no_one_home_range = occupancy.weekday_no_one_home
            
            # Create distributions for sampling the no-one-home period
            weekday_start_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                no_one_home_range.start_hour,
                1.0,
                context={"lower": 0, "upper": no_one_home_range.end_hour})
            
            weekday_end_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                no_one_home_range.end_hour,
                1.0,
                context={"lower": no_one_home_range.start_hour, "upper": 23})
        
        # Presumption distributions for different occupant types
        worker_start_presumption = TranslationRule.RuleSet.normal_distribution_rule()(8.0, 0.5)
        worker_end_presumption = TranslationRule.RuleSet.normal_distribution_rule()(18.0, 0.5)
        hybrid_start_presumption = TranslationRule.RuleSet.normal_distribution_rule()(9.0, 2.0)
        hybrid_end_presumption = TranslationRule.RuleSet.normal_distribution_rule()(17.0, 2.0)
        stayathome_start_presumption = TranslationRule.RuleSet.normal_distribution_rule()(12.0, 3.0)
        stayathome_end_presumption = TranslationRule.RuleSet.normal_distribution_rule()(16.0, 3.0)
        k12_start_presumption = TranslationRule.RuleSet.normal_distribution_rule()(8.0, 0.5)  
        k12_end_presumption = TranslationRule.RuleSet.normal_distribution_rule()(15.0, 0.5)
        college_start_presumption = TranslationRule.RuleSet.normal_distribution_rule()(10.0,3.0)
        college_end_presumption = TranslationRule.RuleSet.normal_distribution_rule()(18.0, 3.0)

        weekday_leaves = []
        weekday_returns = []
        weekday_ranges = []

        for _ in range(5):
            if occupancy.weekday_no_one_home is None:
                daily_range = no_one_home_range
            else:
                start_time,end_time = OccupancyTranslator._sample_single_occupant(weekday_start_dist, weekday_end_dist, wraps_midnight=occupancy.weekday_no_one_home.wraps_midnight)
                daily_range = Occupancy.TimeRange(start_hour=start_time, end_hour=end_time)

            
            weekday_ranges.append(daily_range)
        
            daily_leaves = []
            daily_returns = []
            cnt = 0
            comp = occupancy.household_composition
            # Helper function to sample occupant schedules
            def sample_occupant_schedule(start_presumption, end_presumption, count: int):
                nonlocal cnt
                for _ in range(count):
                    if daily_range.wraps_midnight:
                        # For midnight wraparound, we need different bound handling
                        start_presumption.set_bounds(lower=end_time, upper=23)
                        end_presumption.set_bounds(lower=0, upper=start_time)
                    else:
                        start_presumption.set_bounds(lower = 0, upper=start_time)
                        end_presumption.set_bounds(lower=end_time, upper=23)
                    
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_presumption,
                        end_presumption,
                        wraps_midnight=daily_range.wraps_midnight)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += count

            if comp.fulltime_workers > 0:
                sample_occupant_schedule(worker_start_presumption, worker_end_presumption, 
                                   comp.fulltime_workers)
                
            if comp.hybrid_workers > 0:
                sample_occupant_schedule(hybrid_start_presumption, hybrid_end_presumption, 
                                   comp.hybrid_workers)
                
            if comp.stayathome > 0:
                sample_occupant_schedule(stayathome_start_presumption, stayathome_end_presumption, 
                                   comp.stayathome)
                
            if comp.k12 > 0:
                sample_occupant_schedule(k12_start_presumption, k12_end_presumption, 
                                   comp.k12)
                
            if comp.college_students > 0:
                sample_occupant_schedule(college_start_presumption, college_end_presumption, 
                                   comp.college_students)
                
            if cnt < occupancy.num_occupants:
                needed = occupancy.num_occupants - cnt
                daily_leaves.extend([start_time] * needed)
                daily_returns.extend([end_time] * needed)
            
            weekday_leaves.append(daily_leaves)
            weekday_returns.append(daily_returns)


        weekday_sch = OccupancyTranslator.get_schedule_by_no_one_home(
            weekday_leaves,
            weekday_returns,
            occupancy.num_occupants,
            weekday_ranges
        )
                
        return weekday_sch
    
    def translate_occupancy_presumption_weekend(self) -> list[float]:
        occupancy = self.occupancy
        # Determine if there's a no-one-home period
        if occupancy.weekend_no_one_home is None:
            return [1.0] * 24 * 2  # Everyone stays home all day on weekends
        else:
            no_one_home_range = occupancy.weekend_no_one_home
            
            # Create distributions for sampling the no-one-home period
            weekend_start_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                no_one_home_range.start_hour,
                2.0,
                context={"lower": 0, "upper": no_one_home_range.end_hour})
            
            weekend_end_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                no_one_home_range.end_hour,
                2.0,
                context={"lower": no_one_home_range.start_hour, "upper": 23})
        

            weekend_leaves = []
            weekend_returns = []

            for _ in range(2):
                daily_leaves = []
                daily_returns = []
                for _ in range(occupancy.num_occupants):
                    start_time,end_time = OccupancyTranslator._sample_single_occupant(weekend_start_dist, weekend_end_dist, wraps_midnight=no_one_home_range.wraps_midnight)
                    daily_leaves.append(start_time)
                    daily_returns.append(end_time)
                weekend_leaves.append(daily_leaves)
                weekend_returns.append(daily_returns)

        weekend_sch = OccupancyTranslator._get_schedule_no_constraint(
            weekend_leaves,
            weekend_returns,
            occupancy.num_occupants,
        )
                
        return weekend_sch

    def translate_occupancy_presumption(self) -> list[float]:
        """ Translates occupancy profile into a full week schedule based on presumption method."""
        weekday_schedule = self.translate_occupancy_presumption_weekday()
        weekend_schedule = self.translate_occupancy_presumption_weekend()
        full_week_schedule = weekday_schedule + weekend_schedule
        self.weekly_schedule = full_week_schedule
        return full_week_schedule
    
    def translate_occupancy_sleep_time(self) -> list[tuple[int,int] | None]:
        """
        Translates sleep time ranges into sampled sleep schedules, ensuring sleep periods
        fall entirely within occupied hours.

        Returns:
            List of 7 tuples (sleep_hour, wake_hour) or None for each day of the week
        """
        occupancy = self.occupancy
        if self.weekly_schedule == []:
            self.translate_occupancy_presumption()

        MAX_ATTEMPTS = 100  # Safety limit to prevent infinite loops

        weekday_sleep = []
        weekend_sleep = []

        if occupancy.weekday_sleep_time is None:
            weekday_sleep = [None] * 5
        else:
            sleep_start_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                occupancy.weekday_sleep_time.start_hour,
                0.5,
                context={"lower": 0, "upper": 23})
            sleep_end_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                occupancy.weekday_sleep_time.end_hour,
                0.5,
                context={"lower": 0, "upper": 23})

            wraps_midnight = occupancy.weekday_sleep_time.wraps_midnight

            for day in range(5):
                # Sample sleep times ensuring the entire period is within occupied hours
                valid_sleep_found = False
                for _ in range(MAX_ATTEMPTS):
                    sleep_hour, wake_hour = self._sample_single_occupant(
                        sleep_start_dist,
                        sleep_end_dist,
                        wraps_midnight=wraps_midnight)

                    # Validate entire sleep range is occupied
                    if self._is_time_range_occupied(day, sleep_hour, wake_hour, wraps_midnight):
                        weekday_sleep.append((sleep_hour, wake_hour))
                        valid_sleep_found = True
                        break

                if not valid_sleep_found:
                    # Fallback: use the mean values if no valid sample found
                    weekday_sleep.append((
                        occupancy.weekday_sleep_time.start_hour,
                        occupancy.weekday_sleep_time.end_hour
                    ))

        if occupancy.weekend_sleep_time is None:
            weekend_sleep = [None] * 2
        else:
            sleep_start_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                occupancy.weekend_sleep_time.start_hour,
                0.5,
                context={"lower": 0, "upper": 23})
            sleep_end_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                occupancy.weekend_sleep_time.end_hour,
                0.5,
                context={"lower": 0, "upper": 23})

            wraps_midnight = occupancy.weekend_sleep_time.wraps_midnight

            for day in range(2):
                weekend_day_offset = 5 + day  # Weekend starts after 5 weekdays

                # Sample sleep times ensuring the entire period is within occupied hours
                valid_sleep_found = False
                for _ in range(MAX_ATTEMPTS):
                    sleep_hour, wake_hour = self._sample_single_occupant(
                        sleep_start_dist,
                        sleep_end_dist,
                        wraps_midnight=wraps_midnight)

                    # Validate entire sleep range is occupied
                    if self._is_time_range_occupied(weekend_day_offset, sleep_hour, wake_hour, wraps_midnight):
                        weekend_sleep.append((sleep_hour, wake_hour))
                        valid_sleep_found = True
                        break

                if not valid_sleep_found:
                    # Fallback: use the mean values if no valid sample found
                    weekend_sleep.append((
                        occupancy.weekend_sleep_time.start_hour,
                        occupancy.weekend_sleep_time.end_hour
                    ))

        self.sleep_schedule = weekday_sleep + weekend_sleep
        return self.sleep_schedule
    
    def translate_occupancy_variance_weekday(self) -> list[float]:
        """ Generates a weekday schedule based on occupancy variance using no one home time as mean without further assumptions."""
        occupancy = self.occupancy
        variance = {
            "fulltime_worker": 1.0,
            "hybrid_worker": 2.0,
            "stayathome": 3.0,
            "k12_student": 1.0,
            "college_student": 3.0
        }
        leave_time = occupancy.weekday_no_one_home.start_hour
        return_time = occupancy.weekday_no_one_home.end_hour
        weekday_leaves = []
        weekday_returns = []
        for _ in range(5):
            daily_leaves = []
            daily_returns = []
            cnt = 0
            comp = occupancy.household_composition
            if comp.fulltime_workers > 0:
                for _ in range(comp.fulltime_workers):
                    start_dist = TranslationRule.RuleSet.normal_distribution_rule()(leave_time, variance["fulltime_worker"])
                    end_dist = TranslationRule.RuleSet.normal_distribution_rule()(return_time, variance["fulltime_worker"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.fulltime_workers
            if comp.hybrid_workers > 0:
                for _ in range(comp.hybrid_workers):
                    start_dist = TranslationRule.RuleSet.normal_distribution_rule()(leave_time, variance["hybrid_worker"])
                    end_dist = TranslationRule.RuleSet.normal_distribution_rule()(return_time, variance["hybrid_worker"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.hybrid_workers
            if comp.stayathome > 0:
                for _ in range(comp.stayathome):
                    start_dist = TranslationRule.RuleSet.normal_distribution_rule()(leave_time, variance["stayathome"])
                    end_dist = TranslationRule.RuleSet.normal_distribution_rule()(return_time, variance["stayathome"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.stayathome
            if comp.k12 > 0:
                for _ in range(comp.k12):
                    start_dist = TranslationRule.RuleSet.normal_distribution_rule()(leave_time, variance["k12_student"])
                    end_dist = TranslationRule.RuleSet.normal_distribution_rule()(return_time, variance["k12_student"])
                    start, end = OccupancyTranslator._sample_single_occupant( 
                        start_dist,
                    end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.k12
            if comp.college_students > 0:
                for _ in range(comp.college_students):
                    start_dist = TranslationRule.RuleSet.normal_distribution_rule()(leave_time, variance["college_student"])
                    end_dist = TranslationRule.RuleSet.normal_distribution_rule()(return_time, variance["college_student"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.college_students
            if cnt < occupancy.num_occupants:
                needed = occupancy.num_occupants - cnt
                daily_leaves.extend([leave_time] * needed)
                daily_returns.extend([return_time] * needed)
            weekday_leaves.append(daily_leaves)
            weekday_returns.append(daily_returns)

        weekday_sch = OccupancyTranslator._get_schedule_no_constraint(
            weekday_leaves,
            weekday_returns,
            occupancy.num_occupants
        )
        return weekday_sch

    @staticmethod
    def revise_by_sleep(sleep_weekly:list[tuple[int,int]] , existing_schedule:list[float], value:float) -> list[float]:
        """ Revisions to an existing schedule based on sleep times."""
        revised_schedule = existing_schedule.copy()
        for day in range(7):
            sleep_info = sleep_weekly[day]
            if sleep_info is not None:
                sleep_hour, wake_hour = sleep_info
                if sleep_hour < wake_hour:
                    for h in range(sleep_hour, wake_hour):
                        revised_schedule[day * 24 + h] =value
                else:
                    for h in range(sleep_hour, 24):
                        revised_schedule[day * 24 + h] = value
                    for h in range(0, wake_hour):
                        revised_schedule[day * 24 + h] = value                 
        return revised_schedule

    def get_occupied_active_mask(self) -> list[bool]:
        """ Generates a mask indicating occupied and active hours (True) vs unoccupied or sleep hours (False)."""
        mask = []
        if self.sleep_schedule is None:
            self.translate_occupancy_sleep_time()
        if self.weekly_schedule == []:
            self.translate_occupancy_presumption()
        for h in range(24 * 7):
            if self.weekly_schedule[h] > 0.0:
                mask.append(True)
            else:
                mask.append(False)
        mask = OccupancyTranslator.revise_by_sleep(self.sleep_schedule, mask, False)
        return mask
    
    def get_occupied_mask(self) -> list[bool]:
        """ Generates a mask indicating occupied hours (True) vs unoccupied hours (False)."""
        mask = []
        if self.weekly_schedule == []:
            self.translate_occupancy_presumption()
        for h in range(24 * 7):
            if self.weekly_schedule[h] > 0.0:
                mask.append(True)
            else:
                mask.append(False)
        return mask