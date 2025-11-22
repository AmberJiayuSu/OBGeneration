from stochastic.distribution import Distribution
import model.occupancy as Occuapancy
import prob_model.occupancy as ProbOccupancy
import stochastic.translation_rule as TranslationRule

def sample_single_occupant( start_dist, end_dist) -> tuple[int, int]:
    start_time = start_dist.sample()
    while True:
        end_time = end_dist.sample()
        if end_time > start_time:
            return start_time, end_time

def translate_occupancy_presumption_weekday(occupancy: Occuapancy.Occupancy) -> list[int]:
    noone = True
    if occupancy.weekday_no_one_home.start_hour == occupancy.weekday_no_one_home.end_hour:
        noone = False
    else:
        weekday_start_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(
            occupancy.weekday_no_one_home.start_hour,
            1.0,
            context={"lower": 0, "upper": occupancy.weekday_no_one_home.end_hour})
        
        weekday_end_dist = TranslationRule.RuleSet.int_normal_distribution_rule()(
            occupancy.weekday_no_one_home.end_hour,
            1.0,
            context={"lower": occupancy.weekday_no_one_home.start_hour, "upper": 23})
    
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
    weekday_hard_leave = []
    weekday_hard_return = []

    for _ in range(5):
        if not noone:
            start_time = 23
            end_time = 0
        else:
            start_time = weekday_start_dist.sample()
            end_time = weekday_end_dist.sample()
        weekday_hard_leave.append(start_time)
        weekday_hard_return.append(end_time)
        
        daily_leaves = []
        daily_returns = []
        cnt = 0
        comp = occupancy.household_composition
        if comp.fulltime_workers > 0:
            for _ in range(comp.fulltime_workers):
                worker_start_presumption.set_bounds(upper=start_time)
                worker_end_presumption.set_bounds(lower=end_time)
                start, end = sample_single_occupant(
                    worker_start_presumption,
                    worker_end_presumption)
                daily_leaves.append(start)
                daily_returns.append(end)
            cnt += comp.fulltime_workers
            
        if comp.hybrid_workers > 0:
            for _ in range(comp.hybrid_workers):
                hybrid_start_presumption.set_bounds(upper=start_time)
                hybrid_end_presumption.set_bounds(lower=end_time)
                start, end = sample_single_occupant(
                    hybrid_start_presumption,
                    hybrid_end_presumption)
                daily_leaves.append(start)
                daily_returns.append(end)
            cnt += comp.hybrid_workers
            
        if comp.stayathome > 0:
            for _ in range(comp.stayathome):
                stayathome_start_presumption.set_bounds(upper=start_time)
                stayathome_end_presumption.set_bounds(lower=end_time)
                start, end = sample_single_occupant(
                    stayathome_start_presumption,
                    stayathome_end_presumption)
                daily_leaves.append(start)
                daily_returns.append(end)
            cnt += comp.stayathome
            
        if comp.k12 > 0:
            for _ in range(comp.k12):
                k12_start_presumption.set_bounds(upper=start_time)
                k12_end_presumption.set_bounds(lower=end_time)
                start, end = sample_single_occupant( 
                    k12_start_presumption,
                k12_end_presumption)
                daily_leaves.append(start)
                daily_returns.append(end)
            cnt += comp.k12
            
        if comp.college_students > 0:
            for _ in range(comp.college_students):
                start, end = sample_single_occupant(
                    comp.college_students,
                    college_start_presumption.set_bounds(upper=start_time),
                    college_end_presumption.set_bounds(lower=end_time))
                daily_leaves.append(start)
                daily_returns.append(end)
            cnt += comp.college_students

        if cnt < occupancy.num_occupants:
            needed = occupancy.num_occupants - cnt
            daily_leaves.extend([start_time] * needed)
            daily_returns.extend([end_time] * needed)
        
        weekday_leaves.append(daily_leaves)
        weekday_returns.append(daily_returns)


    weekday_sch = get_schedule(
        weekday_leaves,
        weekday_returns,
        occupancy.num_occupants,
        weekday_hard_leave,
        weekday_hard_return
    )
            
    return weekday_sch

def get_schedule(
    leaves: list[list[int]], 
    returns: list[list[int]], 
    num_occupants: int,
    no_one_home_start_lst: list[int],
    no_one_home_end_lst: list[int]
) -> list[float]:
    """
    Generates the final schedule with strict enforcement of the 'Last Man Standing'
    and 'Absolute Zero' region.
    """
    schedule = []
    ratio = 1.0 / num_occupants
    

    for day_idx in range(len(leaves)):
        day_leaves = leaves[day_idx]
        day_returns = returns[day_idx]
        no_one_home_start = no_one_home_start_lst[day_idx]
        no_one_home_end = no_one_home_end_lst[day_idx]

        print(f"Day {day_idx}: No one home from {no_one_home_start} to {no_one_home_end}")

        for h in range(24):
            # --- REGION 1: ABSOLUTE ZERO (Hard Constraint) ---
            if no_one_home_start <= h < no_one_home_end:
                schedule.append(0.0)
                continue 
            
            # --- REGION 2: CALCULATE RAW PRESENCE ---
            people_present = 0
            for i in range(num_occupants):
                if h < day_leaves[i] or h >= day_returns[i]:
                    people_present += 1

            # --- REGION 3: LAST MAN STANDING (The Fix) ---
            if h < no_one_home_start and people_present == 0:
                
                people_present = 1
            
            if h >= no_one_home_end and people_present == 0:
                people_present = 1
            
            # Calculate fraction
            frac = round(people_present * ratio, 4)
            schedule.append(frac)

    return schedule