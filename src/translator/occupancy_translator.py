from stochastic.distribution import Distribution
import model.occupancy as Occupancy
from stochastic.distribution_config import DistributionConfig
from enum import Enum
import random
from pydantic import BaseModel, Field, ConfigDict
import json
from pathlib import Path



class OccupantRole(Enum):
    DAILY_COMMUTER = "daily_commuter"
    HYBRID_WORKER = "hybrid_worker"
    STAYATHOME = "stayathome"
    K12_OR_DAYCARE = "k12_or_daycare"
    COLLEGE_STUDENT = "college_student"

class RoleAssumption(BaseModel):
    """Defines the behavior for a specific occupant role."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: OccupantRole = Field(..., description="The occupant role.")
    days_away_freq: Distribution = Field(..., description="How many days/week this role leaves home.")
    leave_time: Distribution = Field(..., description="Time of departure.")
    return_time: Distribution = Field(..., description="Time of return.")

class OccupancyAssumptions(BaseModel):
    """Master configuration for household behavioral assumptions."""
    daily_commuter: RoleAssumption
    hybrid_worker: RoleAssumption
    stayathome: RoleAssumption
    k12_or_daycare: RoleAssumption
    college_student: RoleAssumption

    @staticmethod
    def _build_role(d: dict) -> RoleAssumption:
        return RoleAssumption(
            role=OccupantRole(d["role"]),
            days_away_freq=DistributionConfig(**d["days_away_freq"]).build(),
            leave_time=DistributionConfig(**d["leave_time"]).build(),
            return_time=DistributionConfig(**d["return_time"]).build(),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "OccupancyAssumptions":
        data = json.loads(Path(path).read_text())

        return cls(
            daily_commuter=cls._build_role(data["daily_commuter"]),
            hybrid_worker=cls._build_role(data["hybrid_worker"]),
            stayathome=cls._build_role(data["stayathome"]),
            k12_or_daycare=cls._build_role(data["k12_or_daycare"]),
            college_student=cls._build_role(data["college_student"]),
        )

    @classmethod
    def default(cls) -> "OccupancyAssumptions":
        """Returns the standard/default assumptions."""
        return cls(
            daily_commuter=RoleAssumption(
                role = OccupantRole.DAILY_COMMUTER,
                days_away_freq=DistributionConfig(dist_type="constant", params={"value": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 0.5, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 0.5, "int": True}).build()
            ),
            hybrid_worker=RoleAssumption(
                role = OccupantRole.HYBRID_WORKER,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 1, "max": 4}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 0.5, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 18.0, "std": 0.5, "int": True}).build()
            ),
            stayathome=RoleAssumption(
                role = OccupantRole.STAYATHOME,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 0, "max": 2}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 11.0, "std": 2.0, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 13.0, "std": 2.0, "int": True}).build()
            ),
            k12_or_daycare=RoleAssumption(
                role = OccupantRole.K12_OR_DAYCARE,
                days_away_freq=DistributionConfig(dist_type="constant", params={"value": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 8.0, "std": 0.5, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 15.5, "std": 0.5, "int": True}).build()
            ),
            college_student=RoleAssumption(
                role = OccupantRole.COLLEGE_STUDENT,
                days_away_freq=DistributionConfig(dist_type="uniform", params={"min": 3, "max": 5}).build(),
                leave_time=DistributionConfig(dist_type="normal", params={"mean": 10.0, "std": 3.0, "int": True}).build(),
                return_time=DistributionConfig(dist_type="normal", params={"mean": 16.0, "std": 3.0, "int": True}).build()
            )
        )
    

    
    

class WeeklyOccupantAgent:
    def __init__(self, roleAssumption: RoleAssumption):
        self.roleAssumption = roleAssumption
    
    def _get_num_days_away(self) -> int:
        return int(self.roleAssumption.days_away_freq.sample())
        
    def _get_day_away(self):
        all_possible = [0,1,2,3,4]
        n = min(self._get_num_days_away(), len(all_possible))
        return random.sample(all_possible, n)
    
    def sample_weekay(self,no_home_time_ranges: list[Occupancy.TimeRange] ) -> list[Occupancy.TimeRange]:
        schedule = []
        start_dist = self.roleAssumption.leave_time
        end_dist = self.roleAssumption.return_time
        # sample days away based on role assumption, each week indepedent, each person independent
        day_away = self._get_day_away()
        for day in range(5):
            if day in day_away:
                start, end = OccupancyTranslatorUtils._sample_single_range_bound(
                    start_dist,
                    end_dist,
                    no_home_time_ranges[day])
            else:
                start, end = 23,0  # Stays home all day
            schedule.append(Occupancy.TimeRange(start_hour=start, end_hour=end))
        return schedule

    
    
        



class OccupancyTranslator:

    def __init__(self, occupancy: Occupancy.Occupancy, assumptions: OccupancyAssumptions):
        self.occupancy = occupancy
        self.assumptions = assumptions
        self._weekday_occupants_tracker = self._get_occupants_tracker()
        self._defined_occupants_cnt = self._get_defined_occupants_cnt()

    @property 
    def weekday_occupants_tracker(self):
        return self._weekday_occupants_tracker
    
    @property
    def defined_occupants_cnt(self):
        return self._defined_occupants_cnt
    
    def _get_defined_occupants_cnt(self) -> int:
        comp = self.occupancy.household_composition
        return (comp.daily_commuter + comp.hybrid_worker + comp.stayathome +
                comp.k12_or_daycare + comp.college_student)
    
    def _get_occupants_tracker(self):
        comp = self.occupancy.household_composition
        trackers = []
        for _ in range(comp.daily_commuter):
            trackers.append(WeeklyOccupantAgent(self.assumptions.daily_commuter))
        for _ in range(comp.hybrid_worker):
            trackers.append(WeeklyOccupantAgent(self.assumptions.hybrid_worker))
        for _ in range(comp.stayathome):
            trackers.append(WeeklyOccupantAgent(self.assumptions.stayathome))
        for _ in range(comp.k12_or_daycare):
            trackers.append(WeeklyOccupantAgent(self.assumptions.k12_or_daycare))
        for _ in range(comp.college_student):
            trackers.append(WeeklyOccupantAgent(self.assumptions.college_student))
        return trackers
    
    def get_annual_schedule(self) -> tuple[list[list[float]], list[list[bool]]]:
        """ Returns the generated annual occupancy schedule and sleep schedule.
            Each is a list of 53 weeks, each week is a list of 168 hours (or 24 hours for the last week if truncated).
        """
        annual_occupancy_schedule = []
        annual_sleep_schedule = []
        for w in range(53):
            weekly_occupancy, weekly_sleep = self.get_weekly_schedule()
            if w == 53:
                weekly_occupancy = weekly_occupancy[:24]
                weekly_sleep = weekly_sleep[:24]
            annual_occupancy_schedule.append(weekly_occupancy)
            annual_sleep_schedule.append(weekly_sleep)
        return annual_occupancy_schedule, annual_sleep_schedule
    
    
    def get_weekly_schedule(self) -> tuple[list[float], list[bool]]:
        """ Returns the generated weekly occupancy schedule and sleep schedule."""
        occupancy_schedule = self.translate_occupancy_presumption()
        occupancy_mask = OccupancyTranslatorUtils._get_occupancy_mask(occupancy_schedule)
        sleep_schedule = self.translate_occupancy_sleep_time(occupancy_mask)
        sleep_mask = OccupancyTranslatorUtils._get_sleep_mask(sleep_schedule)
        return occupancy_schedule, sleep_mask
    
   


   

    def translate_occupancy_presumption_weekday(self) -> list[float]:
        """ Translates the occupancy profile into a detailed weekday occupancy schedule
        If no one home period is defined, for each occupant, sample their leave and return times for each weekday based on updated presumption distributions using no one home period as bounds. Enforce no one home period for sampled amount of weekdays (between 3 and 5) such that during that period, the household occupancy is strictly zero. Also enforce last man standing outside that period for those weekdays.
        If no one home period is not defined, for each occupant, sample their leave and return times for each weekday based on presumption distributions. Also enforce last man standing for all weekdays.
        """
        occupancy = self.occupancy
        all_possible = [0,1,2,3,4]

        # Get daily no one home ranges for weekdays
        if self.occupancy.weekday_occupancy_pattern.is_always_occupied:
            no_one_home_range = Occupancy.TimeRange(start_hour=23, end_hour=0)  # Zero duration
            no_one_home_ranges = [no_one_home_range] * 5
        else:
            no_one_home_range = occupancy.weekday_occupancy_pattern.no_one_home_interval
            weekday_start_dist = DistributionConfig(dist_type="normal", params={"mean": no_one_home_range.start_hour, "std": 1.0, "int": True,"lower": 0, "upper": no_one_home_range.end_hour}).build()
            weekday_end_dist = DistributionConfig(dist_type="normal", params={"mean": no_one_home_range.end_hour, "std": 1.0, "int": True,"lower": no_one_home_range.start_hour, "upper": 23}).build()
            no_one_home_ranges = []
            for _ in range(5):
                start_time,end_time = OccupancyTranslatorUtils._sample_single_range(weekday_start_dist, weekday_end_dist, wraps_midnight=no_one_home_range.wraps_midnight)
                daily_range = Occupancy.TimeRange(start_hour=start_time, end_hour=end_time)
                no_one_home_ranges.append(daily_range)

        all_occupant_schedules = []
        # Get inidividual occupant schedules for weekdays
        for tracker in self.weekday_occupants_tracker:
            weekday_schedule = tracker.sample_weekay(no_one_home_ranges)
            all_occupant_schedules.append(weekday_schedule)

        # Handle undefined occupants by assigning them the no one home ranges
        if self.defined_occupants_cnt < occupancy.num_occupants:
            # Follow the same no one home ranges for undefined occupants
            all_occupant_schedules.extend([no_one_home_ranges] * (occupancy.num_occupants - self.defined_occupants_cnt))

        # n x 5  ->  5 x n
        all_occupant_schedules = list(map(list, zip(*all_occupant_schedules)))
            
        num_day_enforce = []
        if not self.occupancy.weekday_occupancy_pattern.is_always_occupied:
            num_day_enforce = random.sample(all_possible, self.occupancy.weekday_occupancy_pattern.vacancy_frequency or 5)
        
        schedule = OccupancyTranslator.get_schedule_by_no_one_home(all_occupant_schedules,occupancy.num_occupants, no_one_home_ranges, num_day_enforce)
        return schedule
    


    def translate_occupancy_presumption_weekend(self) -> list[float]:
        # we do not have specific presumption distributions for weekend default occupancy pattern, we solely rely on no one home period if defined
        occupancy = self.occupancy

        if self.occupancy.weekday_occupancy_pattern.is_always_occupied:
            return [1.0] * 24 * 2  # Everyone stays home all day on weekends
        else:
            no_one_home_range = occupancy.weekend_occupancy_pattern.no_one_home_interval
            
            # Create distributions for sampling the no-one-home period
            weekend_start_dist = DistributionConfig(dist_type="normal", params={"mean": no_one_home_range.start_hour, "std": 2.0, "int": True,"lower": 0, "upper": no_one_home_range.end_hour}).build()
            weekend_end_dist = DistributionConfig(dist_type="normal", params={"mean": no_one_home_range.end_hour, "std": 2.0, "int": True,"lower": no_one_home_range.start_hour, "upper": 23}).build()
            
            weekend_leaves = []
            for _ in range(2):
                day_leaves = []
                for _ in range(occupancy.num_occupants):
                    start, end = OccupancyTranslatorUtils._sample_single_range(
                        weekend_start_dist,
                        weekend_end_dist)
                    daily_range = Occupancy.TimeRange(start_hour=start, end_hour=end)
                    day_leaves.append(daily_range)
                weekend_leaves.append(day_leaves)

        weekend_sch = OccupancyTranslator._get_schedule_no_constraint(
            weekend_leaves,
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
    
    
    def translate_occupancy_sleep_time(self, occupancy_mask:list[bool]) -> list[Occupancy.TimeRange | None]:
        """
        Translates sleep time ranges into sampled sleep schedules, ensuring sleep periods
        fall entirely within occupied hours.

        Returns:
            List of 7 tuples (sleep_hour, wake_hour) or None for each day of the week
        """
        occupancy = self.occupancy
        MAX_ATTEMPTS = 100  # Safety limit to prevent infinite loops

        weekday_sleep = []
        weekend_sleep = []

        if occupancy.weekday_occupancy_pattern.sleep_time is None:
            weekday_sleep = [None] * 5
        else:
            sleep_start_dist = DistributionConfig(dist_type="normal", params={"mean": occupancy.weekday_occupancy_pattern.sleep_time.start_hour, "std": 0.5, "int": True,"lower": 0, "upper": 23}).build()
            sleep_end_dist = DistributionConfig(dist_type="normal", params={"mean": occupancy.weekday_occupancy_pattern.sleep_time.end_hour, "std": 0.5, "int": True,"lower": 0, "upper": 23}).build()  

            wraps_midnight = occupancy.weekday_occupancy_pattern.sleep_time.wraps_midnight

            for day in range(5):
                # Sample sleep times ensuring the entire period is within occupied hours
                valid_sleep_found = False
                for _ in range(MAX_ATTEMPTS):
                    sleep_hour, wake_hour = OccupancyTranslatorUtils._sample_single_range(
                        sleep_start_dist,
                        sleep_end_dist,
                        wraps_midnight=wraps_midnight)
                    
                    sleep_range = Occupancy.TimeRange(start_hour=sleep_hour, end_hour=wake_hour)

                    # Validate entire sleep range is occupied
                    if OccupancyTranslatorUtils._is_time_range_occupied(day, sleep_range, occupancy_mask):
                        weekday_sleep.append(sleep_range)
                        valid_sleep_found = True
                        break

                if not valid_sleep_found:
                    # Fallback: use the mean values if no valid sample found
                    weekday_sleep.append(occupancy.weekday_occupancy_pattern.sleep_time)

        if occupancy.weekend_occupancy_pattern.sleep_time is None:
            weekend_sleep = [None] * 2
        else:
            sleep_start_dist = DistributionConfig(dist_type="normal", params={"mean": occupancy.weekend_occupancy_pattern.sleep_time.start_hour, "std": 0.5, "int": True,"lower": 0, "upper": 23}).build()
            sleep_end_dist = DistributionConfig(dist_type="normal", params={"mean": occupancy.weekend_occupancy_pattern.sleep_time.end_hour, "std": 0.5, "int": True,"lower": 0, "upper": 23}).build()
            wraps_midnight = occupancy.weekend_occupancy_pattern.sleep_time.wraps_midnight

            for day in range(2):
                weekend_day_offset = 5 + day  # Weekend starts after 5 weekdays

                # Sample sleep times ensuring the entire period is within occupied hours
                valid_sleep_found = False
                for _ in range(MAX_ATTEMPTS):
                    sleep_hour, wake_hour = OccupancyTranslatorUtils._sample_single_range(
                        sleep_start_dist,
                        sleep_end_dist,
                        wraps_midnight=wraps_midnight)
                    sleep_range = Occupancy.TimeRange(start_hour=sleep_hour, end_hour=wake_hour)

                    # Validate entire sleep range is occupied
                    if OccupancyTranslatorUtils._is_time_range_occupied(weekend_day_offset, sleep_range, occupancy_mask):
                        weekend_sleep.append(sleep_range)
                        valid_sleep_found = True
                        break

                if not valid_sleep_found:
                    # Fallback: use the mean values if no valid sample found
                    weekend_sleep.append(occupancy.weekend_sleep_time)

        self.sleep_schedule = weekday_sleep + weekend_sleep
        return self.sleep_schedule
    
    
    @staticmethod
    def get_schedule_by_no_one_home(
        away_ranges: list[list[Occupancy.TimeRange]],  # m x n
        num_occupants: int, #n
        no_one_home_ranges: list[Occupancy.TimeRange], #m
        num_day_enforce: list[int] = []
    ) -> list[float]:
        """
        Generates the final schedule with strict enforcement of the 'Last Man Standing'
        and 'Absolute Zero' region through no one home period from occupant profile.
        """
        schedule = []
        ratio = 1.0 / num_occupants

        for day_idx in range(len(away_ranges)):
            no_one_home_range = no_one_home_ranges[day_idx]
            weekday_away_range = away_ranges[day_idx]
            for h in range(24):
                # --- REGION 1: ABSOLUTE ZERO (Hard Constraint) ---
                if day_idx in num_day_enforce:
                    if no_one_home_range.contains_hour(h):
                        schedule.append(0.0)
                        continue 

                # -- REGION 2: CALCULATE RAW PRESENCE ---
                people_present = 0
                for i in range(num_occupants):
                    away_range = weekday_away_range[i]
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
            away_ranges: list[list[Occupancy.TimeRange]],  # m x n
            num_occupants: int, #n
        ) -> list[float]:
        """basic schedule generator without hard constraints"""
        schedule = []
        ratio = 1.0 / num_occupants
        for day_idx in range(len(away_ranges)):
            day_away_ranges = away_ranges[day_idx]
            for h in range(24):
                people_present = 0
                for i in range(num_occupants):
                    away_range = day_away_ranges[i]
                    if not away_range.contains_hour(h):
                       people_present += 1
                frac = round(people_present * ratio, 4)
                schedule.append(frac)
        return schedule

    

    
class OccupancyTranslatorUtils:
    @staticmethod
    def _sample_single_range(start_dist, end_dist, wraps_midnight: bool = False) -> tuple[int, int]:
        """
        Samples a single occupant's leave and return times ensuring logical consistency.

        Args:
            start_dist: Distribution for leave time
            end_dist: Distribution for return time
            wraps_midnight: If True, allows return time < leave time (crosses midnight)

        Returns:
            Tuple of (start_time, end_time)
        """
        MAX_ATTEMPTS = 1000
        start_time = start_dist.sample()

        for _ in range(MAX_ATTEMPTS):
            end_time = end_dist.sample()
            if wraps_midnight or end_time > start_time:
                return start_time, end_time

        # Fallback: if bounds make it impossible to satisfy end > start,
        # allow end >= start to avoid infinite loop
        return start_time, end_time   

     # Helper function to sample occupant schedules with updated bounds
    @staticmethod
    def _sample_single_range_bound( start_presumption, end_presumption, daily_noone_range: Occupancy.TimeRange) -> tuple[int, int]:
        if daily_noone_range.wraps_midnight:
            # For midnight wraparound, we need different bound handling
            start_presumption.set_bounds(lower=daily_noone_range.end_hour, upper=23)
            end_presumption.set_bounds(lower=0, upper=daily_noone_range.start_hour)
        else:
            start_presumption.set_bounds(lower = 0, upper=daily_noone_range.start_hour)
            end_presumption.set_bounds(lower=daily_noone_range.end_hour, upper=23)
        
        try:
            start, end = OccupancyTranslatorUtils._sample_single_range(
                start_presumption,
                end_presumption,
                wraps_midnight=daily_noone_range.wraps_midnight)
        except Exception as e:
            print ("Error in sampling occupant schedule with bounds:", e)
            print ("Start presumption:", start_presumption)
            print ("End presumption:", end_presumption)
            raise e
        
        return start, end 
    
    @staticmethod
    def _is_time_range_occupied(day_offset: int, time_range: Occupancy.TimeRange, occupancy_mask:list[bool]) -> bool:
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
        if time_range.wraps_midnight:
            # Check from start_hour to end of day
            for hour in range(time_range.start_hour, 24):
                if not occupancy_mask[day_offset * 24 + hour]:
                    return False
            # Check from start of next day to end_hour
            next_day_offset = (day_offset + 1) % 7
            for hour in range(0, time_range.end_hour):
                if not occupancy_mask[next_day_offset * 24 + hour]:
                    return False
        else:
            # Normal range within same day
            for hour in range(time_range.start_hour, time_range.end_hour):
                if not occupancy_mask[day_offset * 24 + hour]:
                    return False
        return True
    
    @staticmethod
    def _get_occupancy_mask(
        weekly_schedule: list[float]
    ) -> list[bool]:
        """ Generates a mask indicating occupied hours (True) vs unoccupied hours (False)."""
        mask = []
        for h in range(24 * 7):
            if weekly_schedule[h] > 0.0:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
    @staticmethod
    def _get_sleep_mask(sleep_schedule: list[Occupancy.TimeRange | None]) -> list[bool]:
        """ Generates a mask indicating occupied and sleep hours (True) vs unoccupied or active hours (False)."""
        mask = [False] * (24 * 7)
        for d in range(7):
            sleep_info = sleep_schedule[d]
            if sleep_info is not None:
                sleep_hour = sleep_info.start_hour
                wake_hour = sleep_info.end_hour
                if sleep_info.wraps_midnight:
                    for h in range(sleep_hour, 24):
                        mask[d * 24 + h] = True
                    for h in range(0, wake_hour):
                        mask[d * 24 + h] = True
                else:
                    for h in range(sleep_hour, wake_hour):
                        mask[d * 24 + h] = True
        return mask

    @staticmethod
    def get_occupied_active_mask(occupancy_mask:list[bool], sleep_mask:list[bool]) -> list[bool]:
        """ Generates a mask indicating occupied and active hours (True) vs unoccupied or sleep hours (False)."""
        mask = []
        for h in range(24 * 7):
            if occupancy_mask[h] and not sleep_mask[h]:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
    
    @staticmethod
    def revise_by_sleep(sleep_weekly_mask:list[bool] , existing_schedule:list[float], value:float) -> list[float]:
        """ Revisions to an existing schedule based on sleep times. 
            Assume both inputs (sleep_weekly_mask and existing_schedule) should be in the same length."""
        assert len(sleep_weekly_mask) == len(existing_schedule), "sleep mask and schedule must have the same length"
        return [
            value if asleep else v
            for v, asleep in zip(existing_schedule, sleep_weekly_mask)
        ]
    
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

   
   

        