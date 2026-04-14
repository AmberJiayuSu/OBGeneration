"""
rbsa_site.py
------------
Pydantic models representing a single HEMS-RBSA site, assembled from the
suite of HEMS_RBSA CSV files.

Primary source   : HEMS_RBSA_BASE.csv            (one row per site)
Envelope detail  : HEMS_RBSA_Building_Shell_OneLine.csv
HVAC detail      : HEMS_RBSA_Detail_HeatingAndCooling.csv  (one row per unit)
DHW detail       : HEMS_RBSA_Detail_WaterHeating.csv       (one row per unit)
Appliances       : HEMS_RBSA_Appliance_OneLine.csv
Lighting         : HEMS_RBSA_Lighting_OneLine.csv
Usage            : HEMS_RBSA_UsageOneLine.csv

All fields are Optional where the raw data may contain "Not Available",
"Unknown", or similar sentinel strings.  A shared BeforeValidator
(`_clean`) normalises those sentinels to None before Pydantic sees the
value, so downstream code can rely on None meaning "unknown".
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator

# ---------------------------------------------------------------------------
# Sentinel cleaner — applied to every field that may hold a string sentinel
# ---------------------------------------------------------------------------

_INVALID = frozenset(
    {"Not Available", "Unknown", "unknown", "not available", "N/A", "n/a", ""}
)


def _clean(v: object) -> object:
    """Convert RBSA sentinel strings and blank values to None."""
    if v is None:
        return None
    if isinstance(v, float):
        import math
        if math.isnan(v):
            return None
    s = str(v).strip()
    return None if s in _INVALID else v


# Convenience type aliases
MaybeFloat = Annotated[Optional[float], BeforeValidator(_clean)]
MaybeInt   = Annotated[Optional[int],   BeforeValidator(_clean)]
MaybeStr   = Annotated[Optional[str],   BeforeValidator(_clean)]
MaybeBool  = Annotated[Optional[bool],  BeforeValidator(_clean)]


# ---------------------------------------------------------------------------
# Site identity / metadata
# ---------------------------------------------------------------------------

class RBSASiteInfo(BaseModel):
    """Administrative and location metadata for a site."""

    site_id: str
    ee_site_id: MaybeStr = None
    rbsa_data_source: MaybeStr = None

    building_category: MaybeStr = None   # e.g. "Single Family Detached"
    building_type: MaybeStr = None        # e.g. "Single Family"
    home_vintage: MaybeInt = None         # year built

    ownership: MaybeStr = None            # "Own" / "Rent"

    county: MaybeStr = None
    city: MaybeStr = None
    state: MaybeStr = None
    zip: MaybeStr = None

    heating_zone: MaybeInt = None   # RBSA climate heating zone
    cooling_zone: MaybeInt = None   # RBSA climate cooling zone

    # number of rooms and occupants
    qty_rooms: MaybeInt = None
    qty_bedrooms: MaybeInt = None
    qty_bathrooms: MaybeFloat = None   # may be 1.5, 2.5 …
    qty_occupants: MaybeInt = None


# ---------------------------------------------------------------------------
# Geometry 
# ---------------------------------------------------------------------------

class RBSASiteGeometry(BaseModel):
    """Relevant Geometrical Information from Building_Shell_OneLine."""

    #overall
    conditioned_area: MaybeFloat = None    # Conditioned_Area
    conditioned_volume: MaybeFloat = None  # Conditioned_Volume

    #ratios
    conditioned_volume_to_area_ratio: MaybeFloat = None
    wall_area_to_conditioned_area: MaybeFloat = None     # WallArea_to_ConditionedArea
    window_to_wall_ratio: MaybeFloat = None              # Window_to_Wall_Ratio

    # roof / ceiling / attics
    attic_ceiling_area: MaybeFloat = None                     # Attic_Ceiling_Area
    roof_deck_ceiling_area: MaybeFloat = None                # RoofDeck_Ceiling
    sloped_vaulted_ceiling_area: MaybeFloat = None           # SlopedVaulted_Ceiling_Area
    adiabatic_ceiling_area: MaybeFloat = None                # Adiabatic_Ceiling_Area
    total_ceiling_area: MaybeFloat = None                     # Total_Ceiling_Area
    
    # walls
    adiabatic_wall_area: MaybeFloat = None                     # Adiabatic_Wall_Area
    total_wall_area: MaybeFloat = None                     # Total_Wall_Area

    # basement 
    masonry_basement_wall_area: MaybeFloat = None                     # Masonry_Basement_Wall_Area
    basement_floor_area: MaybeFloat = None                     # Basement_Floor_Area

    # floor
    total_floor_area: MaybeFloat = None     # Total_Floor_Area
    crawlspace_floor_area: MaybeFloat = None                     # Crawlspace_Floor_Area
    slab_floor_area: MaybeFloat = None                     # Slab_Floor_Area
    slab_floor_perimeter: MaybeFloat = None                     # Slab_Floor_Perimeter
    floor_over_other_floor_area: MaybeFloat = None                     # Floor_Over_Other_Floor_Area

    #windows / doors / skylights
    window_area: MaybeFloat = None                     # Window_Area
    door_area: MaybeFloat = None                     # Door_Area
    skylight_area: MaybeFloat = None                     # Skylight_Area





# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

class RBSASiteEnvelope(BaseModel):
    """
    Thermal envelope summary from Building_Shell_OneLine.

    Area fields are in ft².
    U-values are in BTU/(hr·ft²·°F)  (IP units as recorded in RBSA).
    UA fields are in BTU/(hr·°F).
    F-value (slab) is in BTU/(hr·ft·°F).
    """

    # --- Ceiling ---
    attic_ceiling_u_value: MaybeFloat = None
    roof_deck_ceiling_u_value: MaybeFloat = None
    sloped_vaulted_ceiling_u_value: MaybeFloat = None
    total_ceiling_u_value: MaybeFloat = None


    # --- Wall ---
    total_wall_u_value: MaybeFloat = None

    # --- Basement ---
    masonry_basement_wall_u_value: MaybeFloat = None
    basement_floor_u_value: MaybeFloat = None

    # --- Floor ---
    crawlspace_floor_u_value: MaybeFloat = None
    slab_floor_f_value: MaybeFloat = None
    total_floor_u_value: MaybeFloat = None
    #floor_ua: MaybeFloat = None

    # --- Windows / doors / skylights ---
    window_u_value: MaybeFloat = None
    door_u_value: MaybeFloat = None
    skylight_u_value: MaybeFloat = None



# ---------------------------------------------------------------------------
# HVAC
# ---------------------------------------------------------------------------

class RBSAThermostat(BaseModel):
    """A single thermostat from HEMS_RBSA_Detail_Thermostat."""

    thermostat_type: MaybeStr = None        # Type — e.g. "Programmable thermostat", "Smart thermostat"
    system_controlled: MaybeStr = None      # System_Controlled — e.g. "Central System", "Zonal"
    room_type: MaybeStr = None              # Room_Type — e.g. "Hallway", "Living Room"


class RBSASiteHVAC(BaseModel):
    """
    HVAC summary for a site from Mechanical_OneLine.
    Captures primary heating/cooling systems, fuels, portable flags,
    and the list of thermostats from Detail_Thermostat.
    """

    primary_heating_system_type: MaybeStr = None   # Primary_Heating_System_Type
    primary_heating_fuel_type: MaybeStr = None     # Primary_Heating_Fuel_Type
    primary_cooling_system_type: MaybeStr = None   # Primary_Cooling_System_Type

    portable_heating: MaybeBool = None             # Portable_Heating_Y/N
    portable_cooling: MaybeBool = None             # Portable_Cooling_Y/N

    thermostats: list[RBSAThermostat] = []         # one entry per Detail_Thermostat row


# ---------------------------------------------------------------------------
# DHW — one entry per water heater (Detail_WaterHeating)
# ---------------------------------------------------------------------------

class RBSADHWUnit(BaseModel):
    """A single water heater from HEMS_RBSA_Detail_WaterHeating."""

    mechanical_id: str

    technology_description: MaybeStr = None   # e.g. "Electric Resistance"
    storage_or_tankless: MaybeStr = None       # "Storage" / "Tankless"
    fuel_type: MaybeStr = None
    size_gallons: MaybeFloat = None
    year_manufactured: MaybeInt = None

    capacity_kw: MaybeFloat = None
    efficiency: MaybeFloat = None              # EF or UEF

    serves_whole_house: MaybeBool = None
    space_is_conditioned: MaybeBool = None
    provided_by_heating_system: MaybeBool = None
    supplemental_solar: MaybeBool = None
    circulator_present: MaybeBool = None



class RBSASiteDHW(BaseModel):
    """DHW summary for a site."""
    units: list[RBSADHWUnit] = []


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

class RBSASiteLighting(BaseModel):
    """Installed lamp inventory from Lighting_OneLine."""

    qty_cfl: MaybeInt = None
    qty_led: MaybeInt = None
    qty_incandescent: MaybeInt = None
    qty_halogen: MaybeInt = None
    qty_linear_fluorescent: MaybeInt = None
    qty_unknown: MaybeInt = None
    qty_other: MaybeInt = None
    total_lamps: MaybeInt = None


# ---------------------------------------------------------------------------
# Appliances
# ---------------------------------------------------------------------------

class RBSASiteAppliances(BaseModel):
    """Appliance inventory from Appliance_OneLine."""

    # Laundry
    washer_total: MaybeInt = None
    washer_energy_star: MaybeInt = None


    dryer_total: MaybeInt = None
    dryer_energy_star: MaybeInt = None
    dryer_electric: MaybeInt = None
    dryer_natural_gas: MaybeInt = None
    dryer_is_heat_pump: MaybeInt = None

    # Refrigeration
    refrigerator_total: MaybeInt = None
    refrigerator_energy_star: MaybeInt = None
    freezer_total: MaybeInt = None
    freezer_energy_star: MaybeInt = None

    # Cooking
    stove_oven_total: MaybeInt = None
    stove_oven_both_electric: MaybeInt = None   # StoveOvenCombo_BothElec
    stove_oven_both_gas: MaybeInt = None        # StoveOvenCombo_BothGas
    stove_oven_stove_gas_oven_elec: MaybeInt = None
    stove_oven_stove_elec_oven_gas: MaybeInt = None

    # Dishwasher
    dishwasher_total: MaybeInt = None
    dishwasher_energy_star: MaybeInt = None

    # other
    dehumidifier_total: MaybeInt = None
    dehumidifier_energy_star: MaybeInt = None
    air_cleaner_total: MaybeInt = None
    computer_laptop_total: MaybeInt = None
    computer_desktop_total: MaybeInt = None
    smart_speaker_total: MaybeInt = None
    game_console_total: MaybeInt = None
    power_strip_total: MaybeInt = None


# ---------------------------------------------------------------------------
# Energy consumption (UsageOneLine)
# ---------------------------------------------------------------------------

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


class RBSASiteEnergy(BaseModel):
    """Metered annual and monthly energy consumption."""

    annual_electric_kwh: MaybeFloat = None
    annual_electric_kbtu: MaybeFloat = None
    annual_gas_therms: MaybeFloat = None
    annual_gas_kbtu: MaybeFloat = None
    annual_wood_cord_kbtu: MaybeFloat = None
    annual_wood_pellet_kbtu: MaybeFloat = None
    annual_propane_kbtu: MaybeFloat = None
    annual_oil_kbtu: MaybeFloat = None

    # Monthly normalised electric usage (kWh)
    january_electric_kwh: MaybeFloat = None
    february_electric_kwh: MaybeFloat = None
    march_electric_kwh: MaybeFloat = None
    april_electric_kwh: MaybeFloat = None
    may_electric_kwh: MaybeFloat = None
    june_electric_kwh: MaybeFloat = None
    july_electric_kwh: MaybeFloat = None
    august_electric_kwh: MaybeFloat = None
    september_electric_kwh: MaybeFloat = None
    october_electric_kwh: MaybeFloat = None
    november_electric_kwh: MaybeFloat = None
    december_electric_kwh: MaybeFloat = None

    # Monthly normalised gas usage (therms)
    january_gas_therms: MaybeFloat = None
    february_gas_therms: MaybeFloat = None
    march_gas_therms: MaybeFloat = None
    april_gas_therms: MaybeFloat = None
    may_gas_therms: MaybeFloat = None
    june_gas_therms: MaybeFloat = None
    july_gas_therms: MaybeFloat = None
    august_gas_therms: MaybeFloat = None
    september_gas_therms: MaybeFloat = None
    october_gas_therms: MaybeFloat = None
    november_gas_therms: MaybeFloat = None
    december_gas_therms: MaybeFloat = None


# ---------------------------------------------------------------------------
# Top-level site model
# ---------------------------------------------------------------------------

class RBSASite(BaseModel):
    """
    Complete representation of a single HEMS-RBSA residential site.

    Constructed by RBSALoader (rbsa_loader.py) which joins across the
    relevant CSV files on SiteID / ee_site_id.
    """

    info: RBSASiteInfo
    geometry: RBSASiteGeometry
    envelope: RBSASiteEnvelope
    hvac: RBSASiteHVAC
    dhw: RBSASiteDHW
    lighting: RBSASiteLighting
    appliances: RBSASiteAppliances
    energy: RBSASiteEnergy
