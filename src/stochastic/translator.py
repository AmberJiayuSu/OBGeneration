from itertools import count
from stochastic.distribution import Distribution
import model.occupancy as Occuapancy
import prob_model.occupancy as ProbOccupancy
import stochastic.translation_rule as TranslationRule


class OccupancyTranslator:

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
        no_one_home_ranges: list[Occuapancy.TimeRange]
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
                    away_range = Occuapancy.TimeRange(start_hour=leave_time, end_hour=return_time)
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
    def translate_occupancy_presumption_weekday(occupancy: Occuapancy.Occupancy) -> list[float]:
        # Determine if there's a no-one-home period
        if occupancy.weekday_no_one_home is None:
            # No one leaves - everyone stays home all day
            no_one_home_range = Occuapancy.TimeRange(start_hour=23, end_hour=0)  # Zero duration
        else:
            no_one_home_range = occupancy.weekday_no_one_home
            
            # Create distributions for sampling the no-one-home period
            weekday_start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(
                no_one_home_range.start_hour,
                1.0,
                context={"lower": 0, "upper": no_one_home_range.end_hour})
            
            weekday_end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(
                no_one_home_range.end_hour,
                1.0,
                context={"lower": no_one_home_range.start_hour, "upper": 23})
        
        # Presumption distributions for different occupant types
        worker_start_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(8.0, 0.5)
        worker_end_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(18.0, 0.5)
        hybrid_start_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(9.0, 2.0)
        hybrid_end_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(17.0, 2.0)
        stayathome_start_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(12.0, 3.0)
        stayathome_end_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(16.0, 3.0)
        k12_start_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(8.0, 0.5)  
        k12_end_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(15.0, 0.5)
        college_start_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(10.0,3.0)
        college_end_presumption = TranslationRule.RuleSet.int_normal_distribution_rule()(18.0, 3.0)

        weekday_leaves = []
        weekday_returns = []
        weekday_ranges = []

        for _ in range(5):
            if occupancy.weekday_no_one_home is None:
                daily_range = no_one_home_range
            else:
                start_time,end_time = OccupancyTranslator._sample_single_occupant(weekday_start_dist, weekday_end_dist, wraps_midnight=occupancy.weekday_no_one_home.wraps_midnight)
                daily_range = Occuapancy.TimeRange(start_hour=start_time, end_hour=end_time)

            
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

    @staticmethod
    def _get_schedule_variance(
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
                    if h < day_leaves[i] or h >= day_returns[i]:
                        people_present += 1
                frac = round(people_present * ratio, 4)
                schedule.append(frac)
        return schedule

    @staticmethod
    def translate_occupancy_variance_weekday(occupancy: Occuapancy.Occupancy) -> list[float]:
        """ Generates a weekday schedule based on occupancy variance using no one home time as mean without further assumptions."""
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
                    start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(leave_time, variance["fulltime_worker"])
                    end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(return_time, variance["fulltime_worker"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.fulltime_workers
            if comp.hybrid_workers > 0:
                for _ in range(comp.hybrid_workers):
                    start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(leave_time, variance["hybrid_worker"])
                    end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(return_time, variance["hybrid_worker"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.hybrid_workers
            if comp.stayathome > 0:
                for _ in range(comp.stayathome):
                    start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(leave_time, variance["stayathome"])
                    end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(return_time, variance["stayathome"])
                    start, end = OccupancyTranslator._sample_single_occupant(
                        start_dist,
                        end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.stayathome
            if comp.k12 > 0:
                for _ in range(comp.k12):
                    start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(leave_time, variance["k12_student"])
                    end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(return_time, variance["k12_student"])
                    start, end = OccupancyTranslator._sample_single_occupant( 
                        start_dist,
                    end_dist)
                    daily_leaves.append(start)
                    daily_returns.append(end)
                cnt += comp.k12
            if comp.college_students > 0:
                for _ in range(comp.college_students):
                    start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(leave_time, variance["college_student"])
                    end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(return_time, variance["college_student"])
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

        weekday_sch = OccupancyTranslator._get_schedule_variance(
            weekday_leaves,
            weekday_returns,
            occupancy.num_occupants
        )
        return weekday_sch



class LightingTranslator:
    # Placeholder for future lighting translation methods
    pass

class ApplianceTranslator:
    # Placeholder for future appliance translation methods
    pass

class HVACTranslator:
    pass

class WindowTranslator:
    pass











