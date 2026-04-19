from typing import NamedTuple


class HouseholdOccupancyFractions(NamedTuple):
    home: float
    sleep: float
    # AWAY is implicit: 1.0 - home - sleep
