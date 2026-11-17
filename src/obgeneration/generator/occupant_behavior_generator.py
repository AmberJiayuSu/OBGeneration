"""One-call generation of a household behavior result from one occupant profile."""

from __future__ import annotations

import numpy as np

import obgeneration.model.equipment as Equipment
from obgeneration.generator.dhw_generator import DHWAssumptions, DHWGenerator
from obgeneration.generator.equipment_generator import EquipmentAssumptions, EquipmentGenerator
from obgeneration.generator.hvac_generator import HVACAssumptions, HVACGenerator
from obgeneration.generator.lighting_generator import LightingAssumptions, LightingGenerator
from obgeneration.generator.occupancy_generator import ClusterAssumptions, OccupancyGenerator
from obgeneration.generator.results import DHWResult, EquipmentResult, OccupantBehaviorResult
from obgeneration.generator.types import HouseholdOccupancyFractions
from obgeneration.generator.window_generator import WindowAssumptions, WindowGenerator
from obgeneration.model.occupant_profile import Occupant


class OccupantBehaviorGenerator:
    """Generate every end use for one `Occupant` on a single RNG stream.

    Call order matches `StochasticScheduleGenerator.generate_schedules`:
    occupancy, then HVAC, then lighting, then equipment, then event-based DHW.
    The window hook runs last and does not draw from `rng`, so lighting stays
    ahead of equipment and a shared seed lines up with that integration.
    HVAC and DHW do not draw from `rng` either.
    """

    @staticmethod
    def generate_result(
        profile: Occupant,
        resolution_mins: int,
        rng: np.random.Generator | int,
        *,
        cluster_assumptions: ClusterAssumptions,
        lighting_assumptions: LightingAssumptions,
        equipment_assumptions: EquipmentAssumptions,
        dhw_assumptions: DHWAssumptions,
        hvac_assumptions: HVACAssumptions,
        window_assumptions: WindowAssumptions,
        active_threshold: float = 0.51,
        away_threshold: float = 0.01,
        min_state_mins: int | None = None,
    ) -> OccupantBehaviorResult:
        """Generate a household result with explicit assumptions.

        `active_threshold` and `away_threshold` match `ScheduleGenerationConfig`:
        a bin is active when home / (home + sleep) is above `active_threshold`,
        and away when home + sleep is at or below `away_threshold`.
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)

        occupancy_result = OccupancyGenerator.generate_result(
            occupancy=profile.occupancy,
            cluster_assumptions=cluster_assumptions,
            resolution_mins=resolution_mins,
            rng=rng,
        )
        hvac_result = HVACGenerator.generate_result(
            hvac=profile.hvac,
            occupancy_states=occupancy_result.occupancy_states,
            assumptions=hvac_assumptions,
            min_state_mins=min_state_mins,
            active_threshold=active_threshold,
            away_threshold=away_threshold,
        )
        lighting_result = LightingGenerator.generate_result(
            lighting=profile.lighting,
            occupancy_states=occupancy_result.occupancy_states,
            lighting_assumptions=lighting_assumptions,
            rng=rng,
            active_threshold=active_threshold,
            away_threshold=away_threshold,
        )
        equipment_result = EquipmentGenerator.generate_result(
            equipment=profile.equipment,
            occupancy_states=occupancy_result.occupancy_states,
            num_occupants=profile.occupancy.num_occupants,
            resolution_mins=resolution_mins,
            equipment_assumptions=equipment_assumptions,
            rng=rng,
        )
        dhw_result = _dhw_from_equipment_events(
            num_occupants=profile.occupancy.num_occupants,
            equipment=profile.equipment,
            occupancy_states=occupancy_result.occupancy_states,
            equipment_result=equipment_result,
            resolution_mins=resolution_mins,
            dhw_assumptions=dhw_assumptions,
        )
        window_result = WindowGenerator.generate_result(
            window=profile.window,
            occupancy_states=occupancy_result.occupancy_states,
            assumptions=window_assumptions,
            resolution_mins=resolution_mins,
            rng=rng,
            hvac=hvac_result,
            active_threshold=active_threshold,
            away_threshold=away_threshold,
        )
        return OccupantBehaviorResult(
            occupancy=occupancy_result,
            lighting=lighting_result,
            equipment=equipment_result,
            dhw=dhw_result,
            hvac=hvac_result,
            window=window_result,
        )

    @staticmethod
    def generate_with_defaults(
        profile: Occupant,
        resolution_mins: int,
        rng: np.random.Generator | int,
        *,
        active_threshold: float = 0.51,
        away_threshold: float = 0.01,
        min_state_mins: int | None = None,
    ) -> OccupantBehaviorResult:
        """Generate a household result using each end use's packaged defaults."""
        return OccupantBehaviorGenerator.generate_result(
            profile=profile,
            resolution_mins=resolution_mins,
            rng=rng,
            cluster_assumptions=ClusterAssumptions.default(),
            lighting_assumptions=LightingAssumptions.default(),
            equipment_assumptions=EquipmentAssumptions.default(resolution_mins),
            dhw_assumptions=DHWAssumptions.default(),
            hvac_assumptions=HVACAssumptions.default(),
            window_assumptions=WindowAssumptions.default(),
            active_threshold=active_threshold,
            away_threshold=away_threshold,
            min_state_mins=min_state_mins,
        )


def _dhw_from_equipment_events(
    *,
    num_occupants: int,
    equipment: Equipment.Equipment,
    occupancy_states: list[list[HouseholdOccupancyFractions]],
    equipment_result: EquipmentResult,
    resolution_mins: int,
    dhw_assumptions: DHWAssumptions,
) -> DHWResult:
    laundry_events = equipment_result.laundry_dhw_event_schedule
    dishwasher_events = equipment_result.dishwasher_dhw_event_schedule
    if laundry_events is None or dishwasher_events is None:
        msg = "Equipment generation did not return DHW event schedules."
        raise ValueError(msg)
    return DHWGenerator.generate_result(
        num_occupants=num_occupants,
        equipment=equipment,
        occupancy_states=occupancy_states,
        laundry_event_schedule=laundry_events,
        dishwasher_event_schedule=dishwasher_events,
        resolution_mins=resolution_mins,
        dhw_assumptions=dhw_assumptions,
    )
