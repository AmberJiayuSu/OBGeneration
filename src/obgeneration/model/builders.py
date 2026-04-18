"""Public simplified builder for external callers.

Each function takes a minimal set of parameters and returns the corresponding
obgeneration input model, with all other fields filled in by defaults.
"""

from collections.abc import Sequence

from obgeneration.model.occupancy import (
    HouseholdComposition,
    MobilityCluster,
    Occupancy,
    OccupantMobilityProfile,
)
from obgeneration.model.lighting import Lighting
from obgeneration.model.equipment import (
    CookingEquipment,
    DishwasherEquipment,
    DishwashingLogic,
    DishwashingPattern,
    Equipment,
    FuelType,
    LaundryEquipment,
    NumberRange,
    RefrigerationEquipment,
)
from obgeneration.model.hvac import HVAC, ThermostatControl


def build_occupancy(
    mobility_clusters: MobilityCluster | Sequence[MobilityCluster],
    num_occupants: int | None = None,
    weekend_clusters: MobilityCluster | Sequence[MobilityCluster] | None = None,
) -> Occupancy:
    """Build an Occupancy profile from mobility cluster(s).
       The whole household can be assigned a single mobility cluster (plus a number of occupants), or each occupant can be assigned their own cluster.
       Optionally, separate weekend clusters can be specified; defaults to same as weekday."""
    if isinstance(mobility_clusters, MobilityCluster):
        if num_occupants is None:
            raise ValueError("num_occupants is required when a single MobilityCluster is given")
        weekday = [mobility_clusters] * num_occupants
    else:
        weekday = mobility_clusters
        num_occupants = len(weekday)

    if weekend_clusters is None:
        weekend = weekday
    elif isinstance(weekend_clusters, MobilityCluster):
        weekend = [weekend_clusters] * num_occupants
    else:
        if len(weekend_clusters) != num_occupants:
            raise ValueError("weekend_clusters must have the same length as mobility_clusters")
        weekend = weekend_clusters

    composition = HouseholdComposition(
        occupants=[
            OccupantMobilityProfile(
                weekday_cluster=wd,
                weekend_cluster=we,
            )
            for wd, we in zip(weekday, weekend)
        ]
    )
    return Occupancy(
        num_occupants=num_occupants,
        household_composition=composition,
    )


def build_lighting(
    if_led: bool,
    when_away: bool | None = False,
    when_daylight_bright: bool | None= True,
) -> Lighting:
    """Build a Lighting profile.

    when_away: lights on even when occupants are away.
    when_daylight_bright: lighting adjusts based on daylight availability.
    """
    if when_away is None:
        when_away = False
    if when_daylight_bright is None:
        when_daylight_bright = True
    return Lighting(
        if_led=if_led,
        when_away=when_away,
        when_daylight_bright=when_daylight_bright,
    )


def build_equipment(
    has_washer: bool,
    has_dryer: bool,
    has_cooking_products: bool,
    has_dishwasher: bool,
    num_refrigerators: int,
    washer_efficient: bool | None = None,
    dryer_efficient: bool | None = None,
    dishwasher_efficient: bool | None = None,
    refrigerator_efficient: bool | None = None,
    laundry_freq_per_week: tuple[int, int] | None = None,
    cooking_freq_per_week: tuple[int, int] | None = None,
    dishwasher_freq_per_week: tuple[int, int] | None = None,
) -> Equipment:
    """Build an Equipment profile from minimal appliance parameters.

    Required flags (has_*) declare appliance presence.
    Efficiency and frequency params are only required when the appliance is present.
    Frequency ranges are (min, max) cycles per week.
    """
    # laundry guards
    if has_washer and washer_efficient is None:
        raise ValueError("washer_efficient is required when has_washer=True")
    if has_washer and laundry_freq_per_week is None:
        raise ValueError("laundry_freq_per_week is required when has_washer=True")
    if has_dryer and dryer_efficient is None:
        raise ValueError("dryer_efficient is required when has_dryer=True")

    # dishwasher guards
    if has_dishwasher and dishwasher_efficient is None:
        raise ValueError("dishwasher_efficient is required when has_dishwasher=True")
    if has_dishwasher and dishwasher_freq_per_week is None:
        raise ValueError("dishwasher_freq_per_week is required when has_dishwasher=True")

    # refrigerator guards
    if num_refrigerators > 0 and refrigerator_efficient is None:
        raise ValueError("refrigerator_efficient is required when num_refrigerators > 0")

    # cooking guards
    if has_cooking_products and cooking_freq_per_week is None:
        raise ValueError("cooking_freq_per_week is required when has_cooking_products=True")

    return Equipment(
        laundry=LaundryEquipment(
            has_washer=has_washer,
            washer_efficient=washer_efficient or False,
            has_dryer=has_dryer,
            dryer_efficient=dryer_efficient or False,
            usage_frequency_per_week=NumberRange(min=laundry_freq_per_week[0], max=laundry_freq_per_week[1]) if has_washer else None,
        ),
        refrigerator=RefrigerationEquipment(
            has_refrigerator=num_refrigerators > 0,
            efficient_refrigerator=refrigerator_efficient or False,
            number_of_refrigerators=num_refrigerators,
        ),
        dishwasher=DishwasherEquipment(
            has_dishwasher=has_dishwasher,
            dishwasher_efficient=dishwasher_efficient or False,
            dishwashing_operational_logic=DishwashingLogic(
                pattern_type=DishwashingPattern.INDEPENDENT_FREQUENCY,
                usage_frequency_per_week=NumberRange(min=dishwasher_freq_per_week[0], max=dishwasher_freq_per_week[1]) if has_dishwasher else None,
            ),
        ),
        cooking_products=CookingEquipment(
            has_cooking_products=has_cooking_products,
            cooking_products_fuel=FuelType.ELECTRIC,
            usage_frequency_per_week=NumberRange(min=cooking_freq_per_week[0], max=cooking_freq_per_week[1]) if has_cooking_products else None,
        ),
    )


def build_hvac(
    has_heating: bool,
    has_cooling: bool,
    heating_setpoint: float | None = None,
    heating_setpoint_sleep: float | None = None,
    heating_setpoint_absent: float | None = None,
    cooling_setpoint: float | None = None,
    cooling_setpoint_sleep: float | None = None,
    cooling_setpoint_absent: float | None = None,
) -> HVAC:
    """Build an HVAC profile from minimal parameters.

    Setpoints are absolute temperatures in °C, only required when the system is present.
    Sleep and absent setpoints default to the active setpoint if not specified.
    """
    if has_heating and heating_setpoint is None:
        raise ValueError("heating_setpoint is required when has_heating=True")
    if has_cooling and cooling_setpoint is None:
        raise ValueError("cooling_setpoint is required when has_cooling=True")

    return HVAC(
        heating=ThermostatControl(
            active_setpoint=heating_setpoint,
            sleep_setpoint=heating_setpoint_sleep or heating_setpoint,
            absent_setpoint=heating_setpoint_absent or heating_setpoint,
        ) if has_heating else ThermostatControl(
            active_setpoint=10.0,
            sleep_setpoint=10.0,
            absent_setpoint=10.0,
        ),
        cooling=ThermostatControl(
            active_setpoint=cooling_setpoint,
            sleep_setpoint=cooling_setpoint_sleep or cooling_setpoint,
            absent_setpoint=cooling_setpoint_absent or cooling_setpoint,
        ) if has_cooling else ThermostatControl(
            active_setpoint=40.0,
            sleep_setpoint=40.0,
            absent_setpoint=40.0,
        ),
    )

def build_cooling(
    has_cooling: bool,
    cooling_setpoint: float | None = None,
    cooling_setpoint_sleep: float | None = None,
    cooling_setpoint_absent: float | None = None,
) -> HVAC:
    """Build a cooling profile from minimal parameters.

    Setpoints are absolute temperatures in °C, only required when cooling is present.
    Sleep and absent setpoints default to the active setpoint if not specified.
    """
    if has_cooling and cooling_setpoint is None:
        raise ValueError("cooling_setpoint is required when has_cooling=True")

    return HVAC(
        cooling=ThermostatControl(
            active_setpoint=cooling_setpoint,
            sleep_setpoint=cooling_setpoint_sleep or cooling_setpoint,
            absent_setpoint=cooling_setpoint_absent or cooling_setpoint,
        ) if has_cooling else ThermostatControl(
            active_setpoint=40.0,
            sleep_setpoint=40.0,
            absent_setpoint=40.0,
        ),
        heating = None,
    )

def build_heating(
    has_heating: bool,
    heating_setpoint: float | None = None,
    heating_setpoint_sleep: float | None = None,
    heating_setpoint_absent: float | None = None,
) -> HVAC:
    """Build a heating profile from minimal parameters.

    Setpoints are absolute temperatures in °C, only required when heating is present.
    Sleep and absent setpoints default to the active setpoint if not specified.
    """
    if has_heating and heating_setpoint is None:
        raise ValueError("heating_setpoint is required when has_heating=True")

    return HVAC(
        heating=ThermostatControl(
            active_setpoint=heating_setpoint,
            sleep_setpoint=heating_setpoint_sleep or heating_setpoint,
            absent_setpoint=heating_setpoint_absent or heating_setpoint,
        ) if has_heating else ThermostatControl(
            active_setpoint=10.0,
            sleep_setpoint=10.0,
            absent_setpoint=10.0,
        ),
        cooling = None,
    )
