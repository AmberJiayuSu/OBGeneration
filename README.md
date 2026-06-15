# Occupant Behavior Generation

`obgeneration` generates annual residential occupant-behavior schedules from a structured occupant profile.

Outputs include:

- occupancy fraction
- lighting usage
- equipment power
- domestic hot water usage
- HVAC setpoints
- window opening fraction

## Structure

```text
src/obgeneration/
├── model/       Pydantic input models
├── generator/   Schedule generators and orchestrator
├── stochastic/  Probability distributions and Markov logic
└── data/        Packaged CSV/JSON runtime assumptions
```

## Installation

```bash
git clone <repo-url>
cd OccupancyGeneration
uv sync
```

For development:

```bash
uv sync --extra dev
```

## Usage

```python
from obgeneration.model import Occupant
from obgeneration.generator import OccupantBehavior

profile = Occupant.from_json_file("test/unit/input/OB_1.json")
annual = OccupantBehavior.to_OB_annual(
    resolution_mins=15,
    occupant_profile=profile,
)

print(annual.num_occupants)
print(len(annual.occupancy_schedule))
```

## Generator Results

The generator convenience methods return structured result models rather than
bare tuples. For example:

```python
from obgeneration.generator import OccupancyGenerator

result = OccupancyGenerator.generate_with_defaults(
    occupancy=profile.occupancy,
    resolution_mins=15,
    rng=42,
)

print(result.peak_value)
print(len(result.schedule))
```

For explicit assumptions and downstream integration, use `generate_result(...)`:

```python
from obgeneration.generator import (
    ClusterAssumptions,
    OccupancyGenerator,
)

result = OccupancyGenerator.generate_result(
    occupancy=profile.occupancy,
    cluster_assumptions=ClusterAssumptions.default(),
    resolution_mins=15,
    rng=42,
)
```
