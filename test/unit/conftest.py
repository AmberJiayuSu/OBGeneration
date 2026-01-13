"""
Shared pytest fixtures.

Fixtures are reusable test setup code. Any fixture defined here
is automatically available in all test files.
"""

import pytest
import numpy as np

from model.occupancy import Occupancy

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def occ_1() -> Occupancy:
    occ_json = """
    {
        "num_occupants": 1,
        "household_composition": {
            "daily_commuter": 1,
            "hybrid_worker": 0,
            "stayathome": 0,
            "k12_or_daycare": 0,
            "college_student": 0
        },
        "weekday_pattern": {
            "is_always_occupied": true
        },
        "weekend_pattern": {
            "is_always_occupied": true
        },
        "sleep_time": {
            "start_hour": 22,
            "end_hour": 6
        }
    }
    """
    return Occupancy.model_validate_json(occ_json)

@pytest.fixture
def occ_2() -> Occupancy:
    occ_json = """
    {
        "num_occupants": 1,
        "household_composition": {
            "daily_commuter": 1,
            "hybrid_worker": 0,
            "stayathome": 0,
            "k12_or_daycare": 0,
            "college_student": 0
        },
        "weekday_pattern": {
            "is_always_occupied": false,
            "away_interval": {
                "start_hour": 8,
                "end_hour": 18
            },
            "num_of_days": 4
        },
        "weekend_pattern": {
            "is_always_occupied": false,
            "away_interval": {
                "start_hour": 16,
                "end_hour": 20
            }
        },
        "sleep_time": {
            "start_hour": 22,
            "end_hour": 6
        }
    }
    """
    return Occupancy.model_validate_json(occ_json)

@pytest.fixture
def occ(request) -> Occupancy:
    """Fixture dispatcher for parameterized tests."""
    return request.getfixturevalue(request.param)