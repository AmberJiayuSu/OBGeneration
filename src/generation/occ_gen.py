import model.occupancy as occ
import generation.stochastic_utils as su

class OccupancyWeeklyScheduleGeneration:
    """
    Class for generating weekly occupancy schedules.
    """
    def __init__(self, occupancy: occ.Occupancy):
        self.occupancy = occupancy

    def generate_schedule(self) -> list[int]:
        raise NotImplementedError("This method should be implemented to generate the weekly occupancy schedule.")
    
    def generate_weekday_schedule(self) -> list[int]:
        schedules = [] [self.occupancy.num_occupants]
        for _ in range (self.occupancy.num_occupants):
            schedule = su.generate_daily_schedule(
                no_one_home_time_range=self.occupancy.weekday_no_one_home,
                sleep_time_range=self.occupancy.weekday_sleep_time
            )
            schedules.append(schedule)

    
    def generate_weekend_schedule(self) -> list[int]:
        raise NotImplementedError("This method should be implemented to generate the weekend occupancy schedule.")
            