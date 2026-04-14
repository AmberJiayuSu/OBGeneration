"""
rbsa_loader.py
--------------
Loads and validates HEMS-RBSA CSV files, constructs RBSASite objects for
every site that passes the key-column validity filter, and saves/loads the
result as an intermediate JSON file.

Validity filter (mirrors weatherfile_creation.ipynb):
  A site is kept only if all key columns are non-null and not a sentinel
  string ("Not Available", "Unknown", …).

Key columns
  Conditioned_Area, Qty_Occupants, Primary_Heating_System_Type,
  Total_Wall_U-Value, Window_U-Value, Window_to_Wall_Ratio

Usage
-----
    from rbsa_loader import RBSALoader

    loader = RBSALoader(data_dir="validation/src/data/HEMS_RBSA")
    sites  = loader.load_all()          # list[RBSASite]
    loader.save(sites, "validation/src/data/rbsa_sites.json")

    # later
    sites = RBSALoader.load_saved("validation/src/data/rbsa_sites.json")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from rbsa_site import (
    RBSASite,
    RBSASiteInfo,
    RBSASiteGeometry,
    RBSASiteEnvelope,
    RBSASiteHVAC,
    RBSAThermostat,
    RBSASiteDHW,
    RBSADHWUnit,
    RBSASiteLighting,
    RBSASiteAppliances,
    RBSASiteEnergy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INVALID = frozenset(
    {"Not Available", "Unknown", "unknown", "not available", "N/A", "n/a", ""}
)

KEY_COLUMNS = [
    "Conditioned_Area",
    "Qty_Occupants",
    "Primary_Heating_System_Type",
    "Total_Wall_U-Value",
    "Window_U-Value",
    "Window_to_Wall_Ratio",
]

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid(val: object) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    return str(val).strip() not in _INVALID


def _get(row: pd.Series, col: str) -> object:
    """Return row[col] or None if the column is missing from this file."""
    return row.get(col, None)


def _yes_no(val: object) -> Optional[bool]:
    """Convert 'Yes'/'No' strings to bool, everything else to None."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() == "yes":
        return True
    if s.lower() == "no":
        return False
    return None


# ---------------------------------------------------------------------------
# Sub-builders — each takes a dict/Series of raw values and returns a model
# ---------------------------------------------------------------------------

def _build_info(site_id: str, base: pd.Series) -> RBSASiteInfo:
    return RBSASiteInfo(
        site_id=site_id,
        ee_site_id=_get(base, "ee_site_id"),
        rbsa_data_source=_get(base, "RBSA Data Source"),
        building_category=_get(base, "Building_Category"),
        building_type=_get(base, "Building_Type"),
        home_vintage=_get(base, "Home_Vintage"),
        ownership=_get(base, "Ownership"),
        county=_get(base, "County"),
        city=_get(base, "City"),
        state=_get(base, "State"),
        zip=str(_get(base, "Zip") or "").split(".")[0] or None,  # strip .0 float artefact
        heating_zone=_get(base, "Heating_Zone"),
        cooling_zone=_get(base, "Cooling_Zone"),
        qty_rooms=_get(base, "Qty_Rooms"),
        qty_bedrooms=_get(base, "Qty_Bedrooms"),
        qty_bathrooms=_get(base, "Qty_Bathrooms"),
        qty_occupants=_get(base, "Qty_Occupants"),
    )


def _build_geometry(shell: pd.Series) -> RBSASiteGeometry:
    return RBSASiteGeometry(
        conditioned_area=_get(shell, "Conditioned_Area"),
        conditioned_volume=_get(shell, "Conditioned_Volume"),
        conditioned_volume_to_area_ratio=_get(shell, "Conditioned_Volume_to_Area_Ratio"),
        wall_area_to_conditioned_area=_get(shell, "Wall_Area_to_Conditioned_Area"),
        window_to_wall_ratio=_get(shell, "Window_to_Wall_Ratio"),
        attic_ceiling_area=_get(shell, "Attic_Ceiling_Area"),
        roof_deck_ceiling_area=_get(shell, "Roof_Deck_Ceiling_Area"),
        sloped_vaulted_ceiling_area=_get(shell, "Sloped_/_Vaulted_(no_attic)_Ceiling_Area"),
        adiabatic_ceiling_area=_get(shell, "Adiabatic_Ceiling_Area"),
        total_ceiling_area=_get(shell, "Total_Ceiling_Area"),
        adiabatic_wall_area=_get(shell, "Adiabatic_Wall_Area"),
        total_wall_area=_get(shell, "Total_Wall_Area"),
        masonry_basement_wall_area=_get(shell, "Masonry_(Basement)_Wall_Area"),
        basement_floor_area=_get(shell, "Basement_Floor_Area"),
        total_floor_area=_get(shell, "Total_Floor_Area"),
        crawlspace_floor_area=_get(shell, "Crawlspace_Floor_Area"),
        slab_floor_area=_get(shell, "Slab_Floor_Area"),
        slab_floor_perimeter=_get(shell, "Slab_Floor_Perimeter"),
        floor_over_other_floor_area=_get(shell, "Floor_over_other_area_Floor_Area"),
        window_area=_get(shell, "Window_Area"),
        door_area=_get(shell, "Door_Area"),
        skylight_area=_get(shell, "Skylight_Area"),
    )


def _build_envelope(shell: pd.Series) -> RBSASiteEnvelope:
    return RBSASiteEnvelope(
        attic_ceiling_u_value=_get(shell, "Attic_Ceiling_U-Value"),
        roof_deck_ceiling_u_value=_get(shell, "Roof_Deck_Ceiling_U-Value"),
        sloped_vaulted_ceiling_u_value=_get(shell, "Sloped_/_Vaulted_(no_attic)_Ceiling_U-Value"),
        total_ceiling_u_value=_get(shell, "Total_Ceiling_U-Value"),
        total_wall_u_value=_get(shell, "Total_Wall_U-Value"),
        masonry_basement_wall_u_value=_get(shell, "Masonry_(Basement)_Wall_U-Value"),
        basement_floor_u_value=_get(shell, "Basement_Floor_U-Value"),
        crawlspace_floor_u_value=_get(shell, "Crawlspace_Floor_U-Value"),
        slab_floor_f_value=_get(shell, "Slab_Floor_F-Value"),
        total_floor_u_value=_get(shell, "Total_Floor_U-Value"),
        window_u_value=_get(shell, "Window_U-Value"),
        door_u_value=_get(shell, "Door_U-Value"),
        skylight_u_value=_get(shell, "Skylight_U-Value"),
    )


def _build_hvac(base: pd.Series, thermostat_rows: pd.DataFrame) -> RBSASiteHVAC:
    thermostats = [
        RBSAThermostat(
            thermostat_type=_get(row, "Type"),
            system_controlled=_get(row, "System_Controlled"),
            room_type=_get(row, "Room_Type"),
        )
        for _, row in thermostat_rows.iterrows()
    ]
    return RBSASiteHVAC(
        primary_heating_system_type=_get(base, "Primary_Heating_System_Type"),
        primary_heating_fuel_type=_get(base, "Primary_Heating_Fuel_Type"),
        primary_cooling_system_type=_get(base, "Primary_Cooling_System_Type"),
        portable_heating=_yes_no(_get(base, "Portable_Heating_Y/N")),
        portable_cooling=_yes_no(_get(base, "Portable_Cooling_Y/N")),
        thermostats=thermostats,
    )


def _build_dhw(dhw_rows: pd.DataFrame) -> RBSASiteDHW:
    units = [
        RBSADHWUnit(
            mechanical_id=str(row["Mechanical_ID"]),
            technology_description=_get(row, "Technology_Description"),
            storage_or_tankless=_get(row, "Storage_Tankless"),
            fuel_type=_get(row, "Fuel_Type"),
            size_gallons=_get(row, "Size"),
            year_manufactured=_get(row, "Year_Manufactured"),
            capacity_kw=_get(row, "Capacity_kW"),
            efficiency=_get(row, "Water_Heater_Efficiency"),
            serves_whole_house=_yes_no(_get(row, "Serves_Whole_House")),
            space_is_conditioned=_yes_no(_get(row, "Space_Is_Conditioned")),
            provided_by_heating_system=_yes_no(_get(row, "Provided_by_Heating_System")),
            supplemental_solar=_yes_no(_get(row, "Supplemental_Solar_Water_Heating")),
            circulator_present=_yes_no(_get(row, "Circulator_Present")),
        )
        for _, row in dhw_rows.iterrows()
    ]
    return RBSASiteDHW(units=units)


def _build_lighting(light: pd.Series) -> RBSASiteLighting:
    return RBSASiteLighting(
        qty_cfl=_get(light, "Qty_Installed_CFL_Lamps"),
        qty_led=_get(light, "Qty_Installed_LED_Lamps"),
        qty_incandescent=_get(light, "Qty_Installed_Incandescent_Lamps"),
        qty_halogen=_get(light, "Qty_Installed_Halogen_Lamps"),
        qty_linear_fluorescent=_get(light, "Qty_Installed_Linear_Fluor._Lamps"),
        qty_unknown=_get(light, "Qty_Installed_Unknown_Lamps"),
        qty_other=_get(light, "Qty_Installed_Other_Lamps"),
        total_lamps=_get(light, "Total_Installed_Lamps"),
    )


def _build_appliances(appl: pd.Series) -> RBSASiteAppliances:
    return RBSASiteAppliances(
        washer_total=_get(appl, "Laundry_Washer_Total"),
        washer_energy_star=_get(appl, "Laundry_Washer_EnergyStar"),
        dryer_total=_get(appl, "Laundry_Dryer_Total"),
        dryer_energy_star=_get(appl, "Laundry_Dryer_EnergyStar"),
        dryer_electric=_get(appl, "Laundry_Dryer_Electric"),
        dryer_natural_gas=_get(appl, "Laundry_Dryer_Natural_Gas"),
        dryer_is_heat_pump=_get(appl, "Laundry_Dryer_IsHeatPump"),
        refrigerator_total=_get(appl, "Refrigerator_Total"),
        refrigerator_energy_star=_get(appl, "Refrigerator_EnergyStar"),
        freezer_total=_get(appl, "Freezer_Total"),
        freezer_energy_star=_get(appl, "Freezer_EnergyStar"),
        stove_oven_total=_get(appl, "Stove_Oven_Total"),
        stove_oven_both_electric=_get(appl, "Stove_Oven_StoveOvenCombo_BothElec"),
        stove_oven_both_gas=_get(appl, "Stove_Oven_StoveOvenCombo_BothGas"),
        stove_oven_stove_gas_oven_elec=_get(appl, "Stove_Oven_StoveOvenCombo_StoveGas_OvenElec"),
        stove_oven_stove_elec_oven_gas=_get(appl, "Stove_Oven_StoveOvenCombo_StoveElec_OvenGas"),
        dishwasher_total=_get(appl, "Dishwasher_Total"),
        dishwasher_energy_star=_get(appl, "Dishwasher_EnergyStar"),
        dehumidifier_total=_get(appl, "Dehumidifier_Total"),
        dehumidifier_energy_star=_get(appl, "Dehumidifier_EnergyStar"),
        air_cleaner_total=_get(appl, "Air_Cleaner_Total"),
        computer_laptop_total=_get(appl, "Computer_Laptop_Total"),
        computer_desktop_total=_get(appl, "Computer_Desktop_Total"),
        smart_speaker_total=_get(appl, "Smart_Speaker_Total"),
        game_console_total=_get(appl, "Game_Console_Total"),
        power_strip_total=_get(appl, "Power_Strip_Total"),
    )


def _build_energy(usage: pd.Series) -> RBSASiteEnergy:
    monthly_elec = {
        f"{m.lower()}_electric_kwh": _get(usage, f"{m} Electric Usage Normalized (kWh)")
        for m in _MONTHS
    }
    monthly_gas = {
        f"{m.lower()}_gas_therms": _get(usage, f"{m} Natural Gas Usage Normalized (therms)")
        for m in _MONTHS
    }
    return RBSASiteEnergy(
        annual_electric_kwh=_get(usage, "Annual Electric Usage (kWh)"),
        annual_electric_kbtu=_get(usage, "Annual Electric Usage (kBtu)"),
        annual_gas_therms=_get(usage, "Annual Gas Usage (therms)"),
        annual_gas_kbtu=_get(usage, "Annual Gas Usage (kBtu)"),
        annual_wood_cord_kbtu=_get(usage, "Annual Wood Cord Usage (kBtu)"),
        annual_wood_pellet_kbtu=_get(usage, "Annual Wood Pellet Usage (kBtu)"),
        annual_propane_kbtu=_get(usage, "Annual Propane Usage (kBtu)"),
        annual_oil_kbtu=_get(usage, "Annual Oil Usage (kBtu)"),
        **monthly_elec,
        **monthly_gas,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RBSALoader:
    """
    Reads all HEMS_RBSA CSV files from data_dir, filters to valid sites,
    and constructs RBSASite objects.
    """

    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self._base      = pd.read_csv(d / "HEMS_RBSA_BASE.csv",                    dtype=str)
        self._shell     = pd.read_csv(d / "HEMS_RBSA_Building_Shell_OneLine.csv",  dtype=str)
        self._thermostat= pd.read_csv(d / "HEMS_RBSA_Detail_Thermostat.csv",       dtype=str)
        self._dhw       = pd.read_csv(d / "HEMS_RBSA_Detail_WaterHeating.csv",     dtype=str)
        self._lighting  = pd.read_csv(d / "HEMS_RBSA_Lighting_OneLine.csv",        dtype=str)
        self._appliances= pd.read_csv(d / "HEMS_RBSA_Appliance_OneLine.csv",       dtype=str)
        self._usage     = pd.read_csv(d / "HEMS_RBSA_UsageOneLine.csv",            dtype=str)

        # Index detail files by SiteID for fast lookup
        self._thermostat_by_site = self._thermostat.groupby("SiteID")
        self._dhw_by_site        = self._dhw.groupby("SiteID")

        # Index one-row-per-site files
        self._shell_idx  = self._shell.set_index("SiteID")
        self._light_idx  = self._lighting.set_index("SiteID")
        self._appl_idx   = self._appliances.set_index("SiteID")
        self._usage_idx  = self._usage.set_index("SiteID")

    def _valid_site_ids(self) -> list[str]:
        """
        Return SiteIDs that pass the key-column validity filter,
        deduplicated by ee_site_id — matching weatherfile_creation.ipynb exactly.
        """
        df = self._base.copy()
        mask = df[KEY_COLUMNS].apply(lambda col: col.map(_is_valid)).all(axis=1)
        kept = df[mask].drop_duplicates(subset="ee_site_id")
        print(f"Kept {len(kept)} / {len(df)} sites after validity filter (deduplicated by ee_site_id)")
        return kept["SiteID"].tolist()

    def _build_site(self, site_id: str) -> RBSASite:
        base  = self._base.set_index("SiteID").loc[site_id]
        shell = self._shell_idx.loc[site_id]  if site_id in self._shell_idx.index  else pd.Series(dtype=str)
        light = self._light_idx.loc[site_id]  if site_id in self._light_idx.index  else pd.Series(dtype=str)
        appl  = self._appl_idx.loc[site_id]   if site_id in self._appl_idx.index   else pd.Series(dtype=str)
        usage = self._usage_idx.loc[site_id]  if site_id in self._usage_idx.index  else pd.Series(dtype=str)

        thermostat_rows = (
            self._thermostat_by_site.get_group(site_id)
            if site_id in self._thermostat_by_site.groups else pd.DataFrame()
        )
        dhw_rows = (
            self._dhw_by_site.get_group(site_id)
            if site_id in self._dhw_by_site.groups else pd.DataFrame()
        )

        return RBSASite(
            info=_build_info(site_id, base),
            geometry=_build_geometry(shell),
            envelope=_build_envelope(shell),
            hvac=_build_hvac(base, thermostat_rows),
            dhw=_build_dhw(dhw_rows),
            lighting=_build_lighting(light),
            appliances=_build_appliances(appl),
            energy=_build_energy(usage),
        )

    def load_all(self) -> list[RBSASite]:
        """Build and return RBSASite objects for all valid sites."""
        site_ids = self._valid_site_ids()
        sites, errors = [], []
        for sid in site_ids:
            try:
                sites.append(self._build_site(sid))
            except Exception as exc:
                errors.append((sid, exc))
        if errors:
            print(f"  {len(errors)} site(s) failed to build:")
            for sid, exc in errors:
                print(f"    {sid}: {exc}")
        print(f"Built {len(sites)} RBSASite objects")
        return sites

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(sites: list[RBSASite], out_path: str | Path) -> None:
        """Serialise sites to a JSON file (list of dicts)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.model_dump() for s in sites]
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Saved {len(sites)} sites -> {out_path}")

    @staticmethod
    def load_saved(path: str | Path) -> list[RBSASite]:
        """Load sites previously saved with RBSALoader.save()."""
        with open(path) as f:
            payload = json.load(f)
        return [RBSASite.model_validate(d) for d in payload]
