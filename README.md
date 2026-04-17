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
src/ob_generation/
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
from ob_generation.model import Occupant
from ob_generation.generator import OccupantBehavior

profile = Occupant.from_json_file("test/unit/input/OB_1.json")
annual = OccupantBehavior.to_OB_annual(
    resolution_mins=15,
    occupant_profile=profile,
)

print(annual.num_occupants)
print(len(annual.occupancy_schedule))
```
