from dataclasses import dataclass
from distribution.distribution import Distribution

@dataclass
class ProbHouseholdComposition:
    fulltime_workers: Distribution
    hybrid_workers: Distribution
    stayathome: Distribution
    k12: Distribution
    college_students: Distribution


@dataclass
class ProbTimeRange:
    start_hour: Distribution
    end_hour: Distribution


@dataclass
class ProbOccupancy:
    num_occupants: Distribution
    household_composition: ProbHouseholdComposition
    weekday_no_one_home: ProbTimeRange
    weekend_no_one_home: ProbTimeRange
    weekday_sleep_time: ProbTimeRange
    weekend_sleep_time: ProbTimeRange