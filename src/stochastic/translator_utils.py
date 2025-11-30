from itertools import count
import numpy as np


class Utils:
    @staticmethod
    def hour_of_day(hour_index: int) -> tuple[int,int]:
        """ Converts a global hour index (0-167) to hour of day (0-23). """
        day = hour_index // 24
        hod = hour_index - (day * 24)
        return day, hod


