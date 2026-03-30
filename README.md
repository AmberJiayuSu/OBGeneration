# Occupant Behavior Generation

Takes a simple JSON occupant profile as input and generates annual household behavior schedules. The occupant profiles are translated into schedules internally using markov chain transitions, probability distributions, and sampling assumptions derived from ATUS and RECS.

Outputs per household: occupancy fractions, lighting, equipment power (W), domestic hot water, HVAC setpoints, and window opening fractions — all at a configurable time resolution.

## Structure

```
src/ob_generation/
├── model/          # Pydantic input models (occupancy, equipment, HVAC, lighting, window)
├── stochastic/     # Probability distributions (normal, uniform, categorical, etc.)
├── generator/      # Schedule generators + main orchestrator (ob_generator.py)
└── data/           # Markov chain transition matrices and activity timing CSVs

test/               # Unit tests and example notebooks
```

## Installation 

```bash
git clone <repo-url>
cd OccupancyGeneration
uv sync
```

Or without uv:
```bash
pip install -e .
```

## Usage

Define an occupant profile as a JSON file (see `test/unit/input/` for examples) and pass it to the generator