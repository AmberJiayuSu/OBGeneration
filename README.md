# Occupant Behavior Generation

`obgeneration` generates annual residential occupant-behavior schedules from a structured occupant profile.

`OccupantBehaviorGenerator` returns one `OccupantBehaviorResult` for a household:

- occupancy fraction, plus the occupancy states later steps use
- lighting usage (fraction × peak W/m²) and a dimming flag
- equipment power (fraction × peak W), plus laundry and dishwasher event schedules
- domestic hot water (fraction × peak m³/s)
- heating and cooling setpoints (°C); either one is absent when that system is absent
- `window`, reserved for an opening-fraction schedule and `None` until that schedule is specified

`OccupantBehavior.to_OB_annual` is the old orchestrator. Its window step calls the same unimplemented hook and stores `None`.

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
cd OBGeneration
uv sync
```

For development:

```bash
uv sync --extra dev
```

## Usage

One profile, one seed, every end use. The call order is occupancy, HVAC, lighting, equipment, then event-based domestic hot water. The window hook runs after that and does not draw from `rng`. HVAC and hot water do not draw from `rng` either, so a shared seed lines up with `StochasticScheduleGenerator.generate_schedules`. `active_threshold` and `away_threshold` default to `0.51` and `0.01`. `result.window` is `None`.

```python
from obgeneration.generator import OccupantBehaviorGenerator
from obgeneration.model import Occupant, Window
from obgeneration.model.builders import (
    build_equipment,
    build_hvac,
    build_lighting,
    build_occupancy,
)
from obgeneration.model.occupancy import MobilityCluster

profile = Occupant(
    occupancy=build_occupancy(
        mobility_clusters=[MobilityCluster.MOSTLY_HOME, MobilityCluster.LONG_DAY_AWAY],
    ),
    lighting=build_lighting(if_led=True, when_away=False, when_daylight_bright=False),
    equipment=build_equipment(
        has_washer=True,
        has_dryer=True,
        has_cooking_products=True,
        has_dishwasher=True,
        num_refrigerators=1,
        washer_efficient=True,
        dryer_efficient=True,
        dishwasher_efficient=True,
        refrigerator_efficient=True,
        laundry_freq_per_week=(1, 2),
        cooking_freq_per_week=(7, 14),
        dishwasher_freq_per_week=(1, 2),
    ),
    hvac=build_hvac(
        has_heating=True,
        has_cooling=True,
        heating_setpoint=21.0,
        heating_setpoint_sleep=18.0,
        heating_setpoint_absent=16.0,
        cooling_setpoint=24.0,
        cooling_setpoint_sleep=26.0,
        cooling_setpoint_absent=28.0,
    ),
    window=Window(),
)

result = OccupantBehaviorGenerator.generate_with_defaults(
    profile,
    resolution_mins=15,
    rng=42,
)

print(result.occupancy.peak_value, len(result.occupancy.annual_schedule))
print(result.lighting.peak_value, result.lighting.dimming_enabled)
print(result.equipment.peak_value)  # watts, not W/m²
print(result.dhw.peak_value)
print(result.hvac.heating.annual_schedule[0])
```

`generate_result(...)` is the same call with explicit assumption objects (`ClusterAssumptions`, `LightingAssumptions`, `EquipmentAssumptions`, `DHWAssumptions`, `HVACAssumptions`, `WindowAssumptions`). Each end use also still has its own `generate_result` / `generate_with_defaults`. `WindowGenerator.generate_result` returns `None`.

`Occupant.from_json_file` loads a profile from JSON. `occupancy` and `lighting` are required. `test/unit/input/OB_1.json`, `OB_2.json`, and `OB_3.json` omit `window`; the loader fills `Window()`, whose season behaviors are all unset.
