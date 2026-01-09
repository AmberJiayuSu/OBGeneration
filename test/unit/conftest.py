"""
Shared pytest fixtures.

Fixtures are reusable test setup code. Any fixture defined here
is automatically available in all test files.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)


