from itertools import count
import numpy as np
from stochastic.distribution import Distribution
import model.occupancy as Occupancy
import stochastic.translation_rule as TranslationRule
from enum import Enum
import random

class OccupantRole(Enum):
    FULLTIME_WORKER = "fulltime_worker"
    HYBRID_WORKER = "hybrid_worker"
    STAYATHOME = "stayathome"
    K12 = "k12"
    COLLEGE_STUDENT = "college_student"

class SingleOccupantTracker:
    def __init__(self, role: OccupantRole):
        self.role = role
        self._num_day_away = self._get_num_days_away()
        self._day_away = self._get_day_away()
    
    @property
    def num_day_away(self) -> int:
        return self._num_day_away
    
    @property
    def day_away(self) -> list[int]:
        return self._day_away
    
    def _get_num_days_away(self) -> int:
        if self.role == OccupantRole.FULLTIME_WORKER:
            return 5
        elif self.role == OccupantRole.HYBRID_WORKER:
            return OccupancyTranslator.role_day_presumption_map["hybrid_worker_days_at_work"].sample()
        elif self.role == OccupantRole.STAYATHOME:
            return OccupancyTranslator.role_day_presumption_map["stayathome_days_out"].sample()
        elif self.role == OccupantRole.COLLEGE_STUDENT:
            return OccupancyTranslator.role_day_presumption_map["college_days_on_campus"].sample()
        elif self.role == OccupantRole.K12:
            return 5
        else:
            return 0
        
    def _get_day_away(self):
        all_possible = [0,1,2,3,4]
        n = min(self.num_day_away, len(all_possible))
        return random.sample(all_possible, n)

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

    # Helper function to sample occupant schedules with updated bounds
    @staticmethod
    def _sample_occupant_schedule_bound( start_presumption, end_presumption, daily_range: Occupancy.TimeRange) -> tuple[int, int]:
        if daily_range.wraps_midnight:
            # For midnight wraparound, we need different bound handling
            start_presumption.set_bounds(lower=daily_range.end_hour, upper=23)
            end_presumption.set_bounds(lower=0, upper=daily_range.start_hour)
        else:
            start_presumption.set_bounds(lower = 0, upper=daily_range.start_hour)
            end_presumption.set_bounds(lower=daily_range.end_hour, upper=23)
        
        start, end = SingleOccupantTracker._sample_single_occupant(
            start_presumption,
            end_presumption,
            wraps_midnight=daily_range.wraps_midnight)
        
        return start, end
    
    @staticmethod
    def _sample_occupant_schedule_mean(start_presumption,end_presumption, daily_range: Occupancy.TimeRange) -> tuple[int, int]:
        if daily_range.wraps_midnight:
            # For midnight wraparound, we need different mean handling
            start_presumption.update_mean((daily_range.end_hour + 23) / 2)
            end_presumption.update_mean((daily_range.start_hour + 0) / 2)
        else:
            start_presumption.update_mean((0 + daily_range.start_hour) / 2)
            end_presumption.update_mean((daily_range.end_hour + 23) / 2)
        
        start, end = SingleOccupantTracker._sample_single_occupant(
            start_presumption,
            end_presumption,
            wraps_midnight=daily_range.wraps_midnight)
        return start, end
    

    def sample_one_day_with_nooneone(self, day:int, occupant_translator:"OccupancyTranslator",dailyrange: Occupancy.TimeRange) -> tuple[int,int]:
        if self.role == OccupantRole.FULLTIME_WORKER:
            start,end = SingleOccupantTracker._sample_occupant_schedule_bound(occupant_translator.role_hour_presumption_map["onsite_worker_start"],occupant_translator.role_hour_presumption_map["onsite_worker_end"], dailyrange)
        elif self.role == OccupantRole.HYBRID_WORKER:
            if day in self.day_away:
                start,end = SingleOccupantTracker._sample_occupant_schedule_bound(occupant_translator.role_hour_presumption_map["onsite_worker_start"],occupant_translator.role_hour_presumption_map["onsite_worker_end"], dailyrange)
            else:
                start, end = 23,0  # Stays home all day 
        elif self.role == OccupantRole.STAYATHOME:
                start, end = dailyrange.start_hour, dailyrange.end_hour  # Stays home all day except no one home period
        elif self.role == OccupantRole.K12:
            start,end = SingleOccupantTracker._sample_occupant_schedule_bound(occupant_translator.role_hour_presumption_map["k12_start"],occupant_translator.role_hour_presumption_map["k12_end"], dailyrange)
        elif self.role == OccupantRole.COLLEGE_STUDENT:
            if day in self.day_away:
                start,end = SingleOccupantTracker._sample_occupant_schedule_bound(occupant_translator.role_hour_presumption_map["college_start"],occupant_translator.role_hour_presumption_map["college_end"], dailyrange)
            else:
                start, end = 23,0  # Stays home all day
        return start, end
    
    def sample_one_day(self, day:int, occupant_translator:"OccupancyTranslator") -> tuple[int,int]:
        if self.role == OccupantRole.FULLTIME_WORKER:
            start,end = SingleOccupantTracker._sample_single_occupant(occupant_translator.role_hour_presumption_map["onsite_worker_start"],occupant_translator.role_hour_presumption_map["onsite_worker_end"])
        elif self.role == OccupantRole.HYBRID_WORKER:
            if day in self.day_away:
                start,end = SingleOccupantTracker._sample_single_occupant(occupant_translator.role_hour_presumption_map["onsite_worker_start"],occupant_translator.role_hour_presumption_map["onsite_worker_end"])
            else:
                start, end = 23,0  # Stays home all day
        elif self.role == OccupantRole.STAYATHOME:
            if day in self.day_away:
                start,end = SingleOccupantTracker._sample_single_occupant(occupant_translator.role_hour_presumption_map["stayathome_start"],occupant_translator.role_hour_presumption_map["stayathome_end"])
            else:
                start, end = 23,0  # Stays home all day
        elif self.role == OccupantRole.K12:
            start,end = SingleOccupantTracker._sample_single_occupant(occupant_translator.role_hour_presumption_map["k12_start"],occupant_translator.role_hour_presumption_map["k12_end"])
        elif self.role == OccupantRole.COLLEGE_STUDENT:
            if day in self.day_away:
                start,end = SingleOccupantTracker._sample_single_occupant(occupant_translator.role_hour_presumption_map["college_start"],occupant_translator.role_hour_presumption_map["college_end"])
            else:
                start, end = 23,0  # Stays home all day
        return start, end
    

        



class OccupancyTranslator:
    role_day_presumption_map = {
        "hybrid_worker_days_at_work": TranslationRule.RuleSet.uniform_distribution_rule()(1,4),
        "stayathome_days_out": TranslationRule.RuleSet.uniform_distribution_rule()(0,2),
        "college_days_on_campus": TranslationRule.RuleSet.uniform_distribution_rule()(3,5)
    }
    days_no_one_home_dist = TranslationRule.RuleSet.uniform_distribution_rule()(3,5)
    role_hour_presumption_map = {
        "onsite_worker_start": TranslationRule.RuleSet.normal_distribution_rule()(8.0, 0.5),
        "onsite_worker_end": TranslationRule.RuleSet.normal_distribution_rule()(18.0, 0.5),
        "k12_start": TranslationRule.RuleSet.normal_distribution_rule()(8.0, 0.5),
        "k12_end": TranslationRule.RuleSet.normal_distribution_rule()(15.0, 0.5),
        "college_start": TranslationRule.RuleSet.normal_distribution_rule()(10.0,3.0),
        "college_end": TranslationRule.RuleSet.normal_distribution_rule()(16.0,3.0),
        "stayathome_start": TranslationRule.RuleSet.normal_distribution_rule()(10.0, 3.0),
        "stayathome_end": TranslationRule.RuleSet.normal_distribution_rule()(14.0, 3.0),
    }

    def __init__(self, occupancy: Occupancy.Occupancy):
        self.occupancy = occupancy
        self.role_map = OccupancyTranslator.role_hour_presumption_map.copy()
        self.weekly_schedule: list[float] = []
        self.sleep_schedule: list[tuple[int,int] | None] = None
        self._weekday_occupants_tracker = self._get_occupants_tracker()

    @property 
    def weekday_occupants_tracker(self):
        return self._weekday_occupants_tracker
    
    def _get_defined_occupants_cnt(self) -> int:
        comp = self.occupancy.household_composition
        return (comp.fulltime_workers + comp.hybrid_workers + comp.stayathome +
                comp.k12 + comp.college_students)
    
    def _get_occupants_tracker(self):
        comp = self.occupancy.household_composition
        trackers = []
        for _ in range(comp.fulltime_workers):
            trackers.append(SingleOccupantTracker(OccupantRole.FULLTIME_WORKER))
        for _ in range(comp.hybrid_workers):
            trackers.append(SingleOccupantTracker(OccupantRole.HYBRID_WORKER))
        for _ in range(comp.stayathome):
            trackers.append(SingleOccupantTracker(OccupantRole.STAYATHOME))
        for _ in range(comp.k12):
            trackers.append(SingleOccupantTracker(OccupantRole.K12))
        for _ in range(comp.college_students):
            trackers.append(SingleOccupantTracker(OccupantRole.COLLEGE_STUDENT))
        return trackers

    def translate_occupancy_presumption_weekday(self) -> list[float]:
        """ Translates the occupancy profile into a detailed weekday occupancy schedule
        If no one home period is defined, for each occupant, sample their leave and return times for each weekday based on updated presumption distributions using no one home period as bounds. Enforce no one home period for sampled amount of weekdays (between 3 and 5) such that during that period, the household occupancy is strictly zero. Also enforce last man standing outside that period for those weekdays.
        If no one home period is not defined, for each occupant, sample their leave and return times for each weekday based on presumption distributions. Also enforce last man standing for all weekdays.
        """
        occupancy = self.occupancy
        cnt = self._get_defined_occupants_cnt()
        weekday_ranges = []
        weekday_leaves = []
        weekday_returns = []
        all_possible = [0,1,2,3,4]
        if self.occupancy.weekday_no_one_home is None:
            no_one_home_range = Occupancy.TimeRange(start_hour=23, end_hour=0)  # Zero duration
            weekday_ranges.extend([no_one_home_range] * 5)
            for i in range(5):
                daily_leaves = []
                daily_returns = []
                for tracker in self.weekday_occupants_tracker:
                    start, end = tracker.sample_one_day(i, self)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                if cnt < occupancy.num_occupants:
                    needed = occupancy.num_occupants - cnt
                    daily_leaves.extend([23] * needed)
                    daily_returns.extend([0] * needed)
                weekday_leaves.append(daily_leaves)
                weekday_returns.append(daily_returns)
            return OccupancyTranslator.get_schedule_by_no_one_home(
                leaves=weekday_leaves,
                returns=weekday_returns,
                num_occupants=occupancy.num_occupants,
                no_one_home_ranges=weekday_ranges,
                num_day_enforce=all_possible)
        else:
            no_one_home_range = occupancy.weekday_no_one_home
            weekday_start_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                no_one_home_range.start_hour,
                1.0,
                context={"lower": 0, "upper": no_one_home_range.end_hour})
            
            weekday_end_dist = TranslationRule.RuleSet.normal_distribution_rule()(
                no_one_home_range.end_hour,
                1.0,
                context={"lower": no_one_home_range.start_hour, "upper": 23})
            for i in range(5):
                daily_leaves = []
                daily_returns = []
                start_time,end_time = SingleOccupantTracker._sample_single_occupant(weekday_start_dist, weekday_end_dist, wraps_midnight=occupancy.weekday_no_one_home.wraps_midnight)
                daily_range = Occupancy.TimeRange(start_hour=start_time, end_hour=end_time)
                weekday_ranges.append(daily_range)
                for tracker in self.weekday_occupants_tracker:
                    start, end = tracker.sample_one_day_with_nooneone(i, self, daily_range)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                if cnt < occupancy.num_occupants:
                    needed = occupancy.num_occupants - cnt
                    daily_leaves.extend([start_time] * needed)
                    daily_returns.extend([end_time] * needed)
                weekday_leaves.append(daily_leaves)
                weekday_returns.append(daily_returns)
            num_day_enforce = self.days_no_one_home_dist.sample()
            return OccupancyTranslator.get_schedule_by_no_one_home(
                leaves=weekday_leaves,
                returns=weekday_returns,
                num_occupants=occupancy.num_occupants,
                no_one_home_ranges=weekday_ranges,
                num_day_enforce=random.sample(all_possible, num_day_enforce))
                
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
                    start_time,end_time = SingleOccupantTracker._sample_single_occupant(weekend_start_dist, weekend_end_dist, wraps_midnight=no_one_home_range.wraps_midnight)
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
                    sleep_hour, wake_hour = SingleOccupantTracker._sample_single_occupant(
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
                    sleep_hour, wake_hour = SingleOccupantTracker._sample_single_occupant(
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
    
    @staticmethod
    def get_schedule_by_no_one_home(
        leaves: list[list[int]], 
        returns: list[list[int]], 
        num_occupants: int,
        no_one_home_ranges: list[Occupancy.TimeRange],
        num_day_enforce: list[int] = []
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
                if day_idx in num_day_enforce:
                    if no_one_home_range.contains_hour(h):
                        schedule.append(0.0)
                        continue 
                #else just continue to region 2 for raw presence calculation
                
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
                # We only enforce this on specified days to avoid over-correction
                if day_idx in num_day_enforce:
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

    def _is_time_range_occupied(self,day_offset: int, start_hour: int, end_hour: int, wraps_midnight: bool = False) -> bool:
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
    
    @staticmethod
    def revise_by_sleep(sleep_weekly_mask:list[bool] , existing_schedule:list[float], value:float) -> list[float]:
        """ Revisions to an existing schedule based on sleep times."""
        revised_schedule = existing_schedule.copy()
        for h in range(24 * 7):
            if sleep_weekly_mask[h]:
                revised_schedule[h] = value
            else:
                revised_schedule[h] = existing_schedule[h]     
        return revised_schedule
    
    @staticmethod
    def revise_by_absence(occupied_weekly_mask:list[bool] , existing_schedule:list[float], value:float) -> list[float]:
        """ Revisions to an existing schedule based on unoccupied times."""
        revised_schedule = existing_schedule.copy()
        for h in range(24 * 7):
            if not occupied_weekly_mask[h]:
                revised_schedule[h] = value
            else:
                revised_schedule[h] = existing_schedule[h]     
        return revised_schedule

    def get_occupied_sleep_mask(self) -> list[bool]:
        """ Generates a mask indicating occupied and sleep hours (True) vs unoccupied or active hours (False)."""
        mask = [False] * (24 * 7)
        if self.sleep_schedule is None:
            self.translate_occupancy_sleep_time()
        if self.weekly_schedule == []:
            self.translate_occupancy_presumption()
        for d in range(7):
            sleep_info = self.sleep_schedule[d]
            if sleep_info is not None:
                sleep_hour, wake_hour = sleep_info
                if sleep_hour < wake_hour:
                    for h in range(sleep_hour, wake_hour):
                        mask[d * 24 + h] = True
                else:
                    for h in range(sleep_hour, 24):
                        mask[d * 24 + h] = True
                    for h in range(0, wake_hour):
                        mask[d * 24 + h] = True
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
    
    def get_occupied_active_mask(self) -> list[bool]:
        """ Generates a mask indicating occupied and active hours (True) vs unoccupied or sleep hours (False)."""
        mask = []
        if self.weekly_schedule == []:
            self.translate_occupancy_presumption()
        if self.sleep_schedule is None:
            self.translate_occupancy_sleep_time()
        sleep_mask = self.get_occupied_sleep_mask()
        occupied_mask = self.get_occupied_mask()
        for h in range(24 * 7):
            if occupied_mask[h] and not sleep_mask[h]:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
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
