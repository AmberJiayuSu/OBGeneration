import numpy as np

from obgeneration.generator.occupancy_generator import OccupancyGenerator, ClusterAssumptions
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.model.occupancy import Occupancy


def _sample_occupancy() -> Occupancy:
    occ_json = """
    {
        "num_occupants": 4,
        "household_composition": {
            "occupants": [
                {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                {"weekday_cluster": "morning_away", "weekend_cluster": "morning_away"},
                {"weekday_cluster": "morning_away", "weekend_cluster": "mostly_home"}
            ]
        }
    }
    """
    return Occupancy.model_validate_json(occ_json)


def _aggregate_states(
    states: list[list[HouseholdOccupancyFractions]],
    factor: int,
) -> list[list[HouseholdOccupancyFractions]]:
    aggregated: list[list[HouseholdOccupancyFractions]] = []
    for week in states:
        assert len(week) % factor == 0
        coarse_week = []
        for i in range(0, len(week), factor):
            block = week[i : i + factor]
            coarse_week.append(
                HouseholdOccupancyFractions(
                    home=sum(x.home for x in block) / factor,
                    sleep=sum(x.sleep for x in block) / factor,
                )
            )
        aggregated.append(coarse_week)
    return aggregated


def test_30_min_generation_matches_15_min_mean_aggregation():
    occupancy = _sample_occupancy()
    assumptions = ClusterAssumptions.default()

    states_15 = OccupancyGenerator(occupancy, assumptions, 15).generate(np.random.default_rng(42))
    states_30 = OccupancyGenerator(occupancy, assumptions, 30).generate(np.random.default_rng(42))

    assert states_30 == _aggregate_states(states_15, factor=2)
    assert OccupancyGenerator.to_occupancy_schedule(states_30) == OccupancyGenerator.to_occupancy_schedule(
        _aggregate_states(states_15, factor=2)
    )


def test_60_min_generation_matches_15_min_mean_aggregation():
    occupancy = _sample_occupancy()
    assumptions = ClusterAssumptions.default()

    states_15 = OccupancyGenerator(occupancy, assumptions, 15).generate(np.random.default_rng(42))
    states_60 = OccupancyGenerator(occupancy, assumptions, 60).generate(np.random.default_rng(42))

    assert states_60 == _aggregate_states(states_15, factor=4)
    assert OccupancyGenerator.to_occupancy_schedule(states_60) == OccupancyGenerator.to_occupancy_schedule(
        _aggregate_states(states_15, factor=4)
    )


def test_5_min_generation_matches_15_min_repetition():
    occupancy = _sample_occupancy()
    assumptions = ClusterAssumptions.default()

    states_15 = OccupancyGenerator(occupancy, assumptions, 15).generate(np.random.default_rng(42))
    states_5 = OccupancyGenerator(occupancy, assumptions, 5).generate(np.random.default_rng(42))

    repeated = [[state for state in week for _ in range(3)] for week in states_15]
    assert states_5 == repeated
