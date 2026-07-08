from collections.abc import Sequence
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from obgeneration.generator.ob_utils import ScheduleUtils
from obgeneration.generator.results import DHWResult
from obgeneration.generator.types import HouseholdOccupancyFractions
import obgeneration.model.equipment as Equipment

WeeklyEventSchedule = Sequence[Sequence[int]]
"""Annual event schedule organized as weeks of integer event ids.

Structure:
- outer sequence length: number of simulated weeks, typically 53
- inner sequence length: number of bins in one week, `7 * 24 * (60 // resolution_mins)`
- value `0`: no appliance DHW event in that timestep
- value `n > 0`: timestep belongs to appliance event `n` within that week

Example for 60-minute resolution:
- one week has 168 bins
- `[0, 0, 1, 1, 0, 2, 2, 2, ...]` means event 1 spans two hourly bins,
  event 2 spans three hourly bins, and other bins have no event draw
"""

WeeklyOccupancyStateSchedule = Sequence[Sequence[HouseholdOccupancyFractions]]
"""Annual occupancy-state schedule organized as weeks.

Structure:
- outer sequence length: number of simulated weeks, typically 53
- inner sequence length: number of bins in one week, `7 * 24 * (60 // resolution_mins)`
- each value is a `HouseholdOccupancyFractions(home, sleep)` tuple

For occupant DHW distribution, the active-at-home ratio is taken from `state.home`, where:
- `0.0` means no occupants are actively at home
- `0.5` means half the household is actively at home
- `1.0` means the full household is actively at home
"""


class DHWAssumptions(BaseModel):
    """DHW usage level assumptions."""

    model_config = ConfigDict(validate_assignment=True)
    hot_water_per_person_per_day: float = Field(
        ..., description="Hot water usage per person per day in liters."
    )
    efficient_washer_per_cycle: float = Field(
        ..., description="Hot water usage per washing machine cycle in liters."
    )
    inefficient_washer_per_cycle: float = Field(
        ..., description="Hot water usage per washing machine cycle in liters."
    )
    efficient_dishwasher_per_cycle: float = Field(
        ..., description="Hot water usage per dishwasher cycle in liters."
    )
    inefficient_dishwasher_per_cycle: float = Field(
        ..., description="Hot water usage per dishwasher cycle in liters."
    )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DHWAssumptions":
        """Load DHW assumptions from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def default(cls) -> "DHWAssumptions":
        """Returns the standard/default assumptions for DHW usage."""
        return cls(
            hot_water_per_person_per_day=56.7812,
            efficient_washer_per_cycle=6.0,
            inefficient_washer_per_cycle=15.0,
            efficient_dishwasher_per_cycle=15.0,
            inefficient_dishwasher_per_cycle=30.0,
        )


def _liters_per_day_to_m3_per_second(liters_per_day: float) -> float:
    """Convert a daily water volume in liters to a constant SI flow rate."""
    cubic_meters_per_day = liters_per_day / 1000.0
    seconds_per_day = 24 * 3600
    return cubic_meters_per_day / seconds_per_day


def _liters_per_timestep_to_m3_per_second(
    liters_per_timestep: float,
    resolution_mins: int,
) -> float:
    """Convert a per-timestep water volume in liters to m^3/s for that bin."""
    cubic_meters = liters_per_timestep / 1000.0
    seconds_per_timestep = resolution_mins * 60
    return cubic_meters / seconds_per_timestep


def _summer_design_day_from_flat_schedule(
    flat_schedule: Sequence[float],
    bins_per_day: int,
) -> list[float] | None:
    """Return the daily slice with the highest average value across the year."""
    max_avg = float("-inf")
    max_day = None
    for day in range(365):
        day_start = day * bins_per_day
        day_schedule = list(flat_schedule[day_start : day_start + bins_per_day])
        day_avg = sum(day_schedule) / bins_per_day
        if day_avg > max_avg:
            max_avg = day_avg
            max_day = day_schedule
    return max_day


def _washer_liters_per_cycle(
    equipment: Equipment.Equipment,
    assumptions: DHWAssumptions,
) -> float:
    """Return washer hot-water liters for one laundry cycle."""
    if not equipment.laundry.has_washer:
        return 0.0
    return (
        assumptions.efficient_washer_per_cycle
        if equipment.laundry.washer_efficient
        else assumptions.inefficient_washer_per_cycle
    )


def _dishwasher_liters_per_cycle(
    equipment: Equipment.Equipment,
    assumptions: DHWAssumptions,
) -> float:
    """Return dishwasher hot-water liters for one dishwasher cycle."""
    if not equipment.dishwasher.has_dishwasher:
        return 0.0
    return (
        assumptions.efficient_dishwasher_per_cycle
        if equipment.dishwasher.dishwasher_efficient
        else assumptions.inefficient_dishwasher_per_cycle
    )


def _occupants_dhw_liters_per_day(num_occupants: int, assumptions: DHWAssumptions) -> float:
    """Return total daily occupant-driven DHW volume for the household."""
    return num_occupants * assumptions.hot_water_per_person_per_day


class DHWFlatGenerator:
    """Legacy DHW generator using daily cycle counts flattened uniformly across each day."""

    def __init__(
        self,
        dhw_assumptions: DHWAssumptions,
        equipment: Equipment.Equipment,
        num_occupants: int,
        laundry_cycles_per_day: Sequence[Sequence[int]],
        dishwasher_cycles_per_day: Sequence[Sequence[int]],
        resolution_mins: int = 15,
    ):
        """Initialize the legacy daily-total DHW generator."""
        self.dhw_assumptions = dhw_assumptions
        self.equipment = equipment
        self.num_occupants = num_occupants
        self.laundry_cycles_per_day = laundry_cycles_per_day
        self.dishwasher_cycles_per_day = dishwasher_cycles_per_day
        self.resolution_mins = resolution_mins

    @staticmethod
    def generate_result(
        num_occupants: int,
        equipment: Equipment.Equipment,
        laundry_cycles_per_day: Sequence[Sequence[int]],
        dishwasher_cycles_per_day: Sequence[Sequence[int]],
        resolution_mins: int,
        dhw_assumptions: DHWAssumptions,
    ) -> DHWResult:
        """Generate DHW results from daily cycle counts using the flat legacy path."""
        generator = DHWFlatGenerator(
            dhw_assumptions=dhw_assumptions,
            equipment=equipment,
            num_occupants=num_occupants,
            laundry_cycles_per_day=laundry_cycles_per_day,
            dishwasher_cycles_per_day=dishwasher_cycles_per_day,
            resolution_mins=resolution_mins,
        )
        flow_rate, annual_schedule = generator.dhw_annual_schedule()
        flattened_schedule = ScheduleUtils.flatten_schedule(annual_schedule)
        bins_per_day = 1440 // resolution_mins
        max_day = _summer_design_day_from_flat_schedule(flattened_schedule, bins_per_day)
        return DHWResult(
            peak_value=flow_rate,
            annual_schedule=flattened_schedule,
            summer_design_day_schedule=max_day,
            winter_design_day_schedule=max_day,
        )

    @staticmethod
    def generate_with_defaults(
        num_occupants: int,
        equipment: Equipment.Equipment,
        laundry_cycles_per_day: Sequence[Sequence[int]],
        dishwasher_cycles_per_day: Sequence[Sequence[int]],
        resolution_mins: int,
    ) -> DHWResult:
        """Generate flat-path DHW results using default DHW assumptions."""
        return DHWFlatGenerator.generate_result(
            num_occupants=num_occupants,
            equipment=equipment,
            laundry_cycles_per_day=laundry_cycles_per_day,
            dishwasher_cycles_per_day=dishwasher_cycles_per_day,
            resolution_mins=resolution_mins,
            dhw_assumptions=DHWAssumptions.default(),
        )

    def dhw_annual_schedule(self) -> tuple[float, list[list[float]]]:
        """Build a normalized annual schedule by spreading each day's liters evenly."""
        annual_schedule = []
        annual_daily = []
        for week_index in range(len(self.laundry_cycles_per_day)):
            weekly_daily_dhw = self.dhw_weekly_schedule(
                self.laundry_cycles_per_day[week_index],
                self.dishwasher_cycles_per_day[week_index],
            )
            annual_daily.append(weekly_daily_dhw)

        max_flow = max(max(week) for week in annual_daily)
        fraction = [[day / max_flow for day in week] for week in annual_daily] if max_flow > 0 else annual_daily
        flow_rate = _liters_per_day_to_m3_per_second(max_flow)

        bins_per_day = 24 * (60 // self.resolution_mins)
        for week_fraction in fraction:
            week_schedule = []
            for day_fraction in week_fraction:
                week_schedule.extend([day_fraction] * bins_per_day)
            annual_schedule.append(week_schedule)
        return flow_rate, annual_schedule

    def dhw_weekly_schedule(
        self,
        weekly_laundry_cycles_per_day: Sequence[int],
        weekly_dishwasher_cycles_per_day: Sequence[int],
    ) -> list[float]:
        """Return one week of daily DHW totals in liters for the flat legacy path."""
        daily_occupants_dhw = self.occupants_dhw()
        daily_laundry_dhw = self.weekly_laundry_dhw(weekly_laundry_cycles_per_day)
        daily_dishwasher_dhw = self.weekly_dishwasher_dhw(weekly_dishwasher_cycles_per_day)
        return [
            daily_occupants_dhw + laundry + dishwasher
            for laundry, dishwasher in zip(daily_laundry_dhw, daily_dishwasher_dhw)
        ]

    def occupants_dhw(self) -> float:
        """Return the household's occupant-driven DHW liters for one day."""
        return _occupants_dhw_liters_per_day(self.num_occupants, self.dhw_assumptions)

    def weekly_laundry_dhw(self, weekly_laundry_cycles_per_day: Sequence[int]) -> list[float]:
        """Convert daily washer cycle counts into daily DHW liters."""
        if not self.equipment.laundry.has_washer:
            return [0.0] * len(weekly_laundry_cycles_per_day)
        per_cycle = _washer_liters_per_cycle(self.equipment, self.dhw_assumptions)
        return [day_cycles * per_cycle for day_cycles in weekly_laundry_cycles_per_day]

    def weekly_dishwasher_dhw(self, weekly_dishwasher_cycles_per_day: Sequence[int]) -> list[float]:
        """Convert daily dishwasher cycle counts into daily DHW liters."""
        if not self.equipment.dishwasher.has_dishwasher:
            return [0.0] * len(weekly_dishwasher_cycles_per_day)
        per_cycle = _dishwasher_liters_per_cycle(self.equipment, self.dhw_assumptions)
        return [day_cycles * per_cycle for day_cycles in weekly_dishwasher_cycles_per_day]


class DHWGenerator:
    """Event-based DHW generator using appliance event schedules directly.

    `laundry_event_schedule` and `dishwasher_event_schedule` are annual weekly
    schedules of integer event ids, not daily counts.
    """

    def __init__(
        self,
        dhw_assumptions: DHWAssumptions,
        equipment: Equipment.Equipment,
        num_occupants: int,
        occupancy_states: WeeklyOccupancyStateSchedule,
        laundry_event_schedule: WeeklyEventSchedule,
        dishwasher_event_schedule: WeeklyEventSchedule,
        resolution_mins: int = 15,
    ):
        """Initialize event-based DHW generation.

        Args:
            occupancy_states: Weekly household occupancy states. Each inner list
                is one week and should align exactly with the event schedules.
                Occupant DHW is distributed within each day in proportion to
                `state.home` in each timestep.
            laundry_event_schedule: Weekly integer-id schedule for washer-related
                DHW events across the full year. Each inner list is one week.
                Bins with `0` mean no laundry DHW draw. Bins with the same
                positive integer belong to the same laundry event and split one
                cycle's DHW volume uniformly across those bins.
            dishwasher_event_schedule: Same structure as
                `laundry_event_schedule`, but for dishwasher events.
        """
        self.dhw_assumptions = dhw_assumptions
        self.equipment = equipment
        self.num_occupants = num_occupants
        self.occupancy_states = occupancy_states
        self.laundry_event_schedule = laundry_event_schedule
        self.dishwasher_event_schedule = dishwasher_event_schedule
        self.resolution_mins = resolution_mins

    @staticmethod
    def generate_result(
        num_occupants: int,
        equipment: Equipment.Equipment,
        occupancy_states: WeeklyOccupancyStateSchedule,
        laundry_event_schedule: WeeklyEventSchedule,
        dishwasher_event_schedule: WeeklyEventSchedule,
        resolution_mins: int,
        dhw_assumptions: DHWAssumptions,
    ) -> DHWResult:
        """Generate annual DHW from explicit weekly event-id schedules."""
        generator = DHWGenerator(
            dhw_assumptions=dhw_assumptions,
            equipment=equipment,
            num_occupants=num_occupants,
            occupancy_states=occupancy_states,
            laundry_event_schedule=laundry_event_schedule,
            dishwasher_event_schedule=dishwasher_event_schedule,
            resolution_mins=resolution_mins,
        )
        flow_rate, annual_schedule = generator.dhw_annual_schedule()
        flattened_schedule = ScheduleUtils.flatten_schedule(annual_schedule)
        bins_per_day = 1440 // resolution_mins
        max_day = _summer_design_day_from_flat_schedule(flattened_schedule, bins_per_day)
        return DHWResult(
            peak_value=flow_rate,
            annual_schedule=flattened_schedule,
            summer_design_day_schedule=max_day,
            winter_design_day_schedule=max_day,
        )

    @staticmethod
    def generate_with_defaults(
        num_occupants: int,
        equipment: Equipment.Equipment,
        occupancy_states: WeeklyOccupancyStateSchedule,
        laundry_event_schedule: WeeklyEventSchedule,
        dishwasher_event_schedule: WeeklyEventSchedule,
        resolution_mins: int,
    ) -> DHWResult:
        """Generate annual DHW from event schedules using default assumptions."""
        return DHWGenerator.generate_result(
            num_occupants=num_occupants,
            equipment=equipment,
            occupancy_states=occupancy_states,
            laundry_event_schedule=laundry_event_schedule,
            dishwasher_event_schedule=dishwasher_event_schedule,
            resolution_mins=resolution_mins,
            dhw_assumptions=DHWAssumptions.default(),
        )

    def dhw_annual_schedule(self) -> tuple[float, list[list[float]]]:
        """Return normalized annual DHW schedule from weekly event schedules.

        The returned schedule remains organized as weeks. Each event id in the
        input schedules is converted into a constant per-bin draw across the bins
        occupied by that event.
        """
        annual_schedule_liters = []
        for week_index in range(len(self.laundry_event_schedule)):
            weekly_schedule_liters = self.dhw_weekly_flow_schedule(
                self.occupancy_states[week_index],
                self.laundry_event_schedule[week_index],
                self.dishwasher_event_schedule[week_index],
            )
            annual_schedule_liters.append(weekly_schedule_liters)

        max_liters_per_timestep = max(max(week) for week in annual_schedule_liters)
        normalized_schedule = (
            [
                [value / max_liters_per_timestep for value in week]
                for week in annual_schedule_liters
            ]
            if max_liters_per_timestep > 0
            else annual_schedule_liters
        )
        flow_rate = _liters_per_timestep_to_m3_per_second(
            max_liters_per_timestep,
            self.resolution_mins,
        )
        return flow_rate, normalized_schedule

    def dhw_weekly_flow_schedule(
        self,
        weekly_occupancy_states: Sequence[HouseholdOccupancyFractions],
        weekly_laundry_event_schedule: Sequence[int],
        weekly_dishwasher_event_schedule: Sequence[int],
    ) -> list[float]:
        """Generate one week's DHW liters per timestep from weekly event ids.

        Example inner-week structure at 15-minute resolution:
        - week length is 672 bins
        - occupancy states with `state.home` like `[0.0, 0.0, 0.5, 1.0, ...]`
          mean occupant DHW for that day is distributed in proportion to those
          active-at-home weights
        - `[0, 0, 5, 5, 5, 0, ...]` means laundry event id 5 occupies three
          consecutive 15-minute bins, so one cycle's liters are split evenly
          across those three bins
        """
        if len(weekly_occupancy_states) != len(weekly_laundry_event_schedule):
            raise ValueError(
                "Occupancy states and laundry event schedules must have the same weekly length."
            )
        if len(weekly_laundry_event_schedule) != len(weekly_dishwasher_event_schedule):
            raise ValueError(
                "Laundry and dishwasher event schedules must have the same weekly length."
            )

        bins_per_day = 24 * (60 // self.resolution_mins)
        if len(weekly_laundry_event_schedule) % bins_per_day != 0:
            raise ValueError(
                f"Weekly event schedule length {len(weekly_laundry_event_schedule)} is not divisible by bins per day {bins_per_day}."
            )

        weekly_schedule = [0.0] * len(weekly_laundry_event_schedule)
        num_days = len(weekly_laundry_event_schedule) // bins_per_day

        self._apply_daily_occupant_background(
            weekly_schedule,
            weekly_occupancy_states,
            num_days,
        )
        self._apply_event_schedule(
            weekly_schedule,
            weekly_laundry_event_schedule,
            _washer_liters_per_cycle(self.equipment, self.dhw_assumptions),
        )
        self._apply_event_schedule(
            weekly_schedule,
            weekly_dishwasher_event_schedule,
            _dishwasher_liters_per_cycle(self.equipment, self.dhw_assumptions),
        )
        return weekly_schedule

    def _apply_daily_occupant_background(
        self,
        weekly_schedule: list[float],
        weekly_occupancy_states: Sequence[HouseholdOccupancyFractions],
        num_days: int,
    ) -> None:
        """Distribute each day's occupant DHW across bins using `state.home` weights.

        For each day, the household's daily occupant-driven DHW volume is spread
        across that day's bins in proportion to the active-at-home ratio stored
        in `state.home`. If the full day has zero active-at-home weight, the
        daily total falls back to a uniform distribution across all bins.
        """
        bins_per_day = 24 * (60 // self.resolution_mins)
        daily_total = _occupants_dhw_liters_per_day(
            self.num_occupants,
            self.dhw_assumptions,
        )
        for day_index in range(num_days):
            day_start = day_index * bins_per_day
            day_end = day_start + bins_per_day
            day_weights = [
                max(0.0, float(state.home))
                for state in weekly_occupancy_states[day_start:day_end]
            ]
            total_weight = sum(day_weights)
            if total_weight > 0:
                for offset, weight in enumerate(day_weights):
                    weekly_schedule[day_start + offset] += daily_total * weight / total_weight
            else:
                per_bin = daily_total / bins_per_day
                for bin_index in range(day_start, day_end):
                    weekly_schedule[bin_index] += per_bin

    def _apply_event_schedule(
        self,
        weekly_schedule: list[float],
        weekly_event_schedule: Sequence[int],
        liters_per_cycle: float,
    ) -> None:
        """Add event-based appliance DHW by splitting each cycle over its bins.

        Each positive event id identifies one appliance cycle. All bins with the
        same id receive an equal share of `liters_per_cycle`, which produces a
        constant draw over that event's duration.
        """
        if liters_per_cycle == 0:
            return
        if len(weekly_event_schedule) != len(weekly_schedule):
            raise ValueError(
                f"Event schedule length {len(weekly_event_schedule)} does not match expected weekly length {len(weekly_schedule)}."
            )

        event_bins: dict[int, list[int]] = {}
        for bin_index, event_id in enumerate(weekly_event_schedule):
            if event_id > 0:
                event_bins.setdefault(int(event_id), []).append(bin_index)

        for bins in event_bins.values():
            per_bin = liters_per_cycle / len(bins)
            for bin_index in bins:
                weekly_schedule[bin_index] += per_bin
