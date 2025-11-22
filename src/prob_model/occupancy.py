from dataclasses import dataclass
from stochastic.distribution import Distribution

@dataclass
class SingleOccupant:
    occupation_type: str  # 'fulltime_worker', 'hybrid_worker', 'stayathome', 'k12_student', 'college_student'
    weekday_leave_time: Distribution
    weekday_return_time: Distribution
    weekend_leave_time: Distribution
    weekend_return_time: Distribution

@dataclass
class HouseholdOccupancy:
    num_occupants: int
    occupants: list[SingleOccupant]
    weekday_sleep_start: Distribution
    weekday_sleep_end: Distribution
    weekend_sleep_start: Distribution
    weekend_sleep_end: Distribution