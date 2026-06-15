"""Build model and behavior input records for RBSA III sites."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RBSA_III_ModelInput:
    """Cleaned, prejoined RBSA III facts for one modeled building."""

    building_id: str
    site_id: str
    #ee_site_id: str | None = None

    conditioned_area_ft2: float | None = None
    conditioned_volume_ft3: float | None = None
    average_height_ft: float | None = None
    conditioned_volume_to_area_ratio_ft: float | None = None
    total_building_levels: int | None = None
    inferred_total_building_levels: int | None = None
    effective_total_building_levels: int | None = None
    footprint_area_ft2: float | None = None
    footprint_perimeter_ft: float | None = None
    has_conditioned_basement: bool | None = None
    footprint_based_conditioned_area_ft2: float | None = None
    conditioned_area_difference_ft2: float | None = None
    conditioned_area_absolute_difference_ft2: float | None = None
    conditioned_area_difference_pct: float | None = None
    footprint_conditioned_area_issue: bool | None = None
    total_wall_u_value_missing_issue: bool = False
    window_to_wall_ratio_missing_issue: bool = False
    metered_data_high_missing_rate: bool = False
    has_data_quality_issue: bool = False

    window_to_wall_ratio: float | None = None
    total_wall_u_value_ip: float | None = None
    total_ceiling_u_value_ip: float | None = None
    total_floor_u_value_ip: float | None = None
    window_u_value_ip: float | None = None
    ach50: float | None = None
    pct_windows_facing_north: float | None = None
    pct_windows_facing_northeast: float | None = None
    pct_windows_facing_east: float | None = None
    pct_windows_facing_southeast: float | None = None
    pct_windows_facing_south: float | None = None
    pct_windows_facing_southwest: float | None = None
    pct_windows_facing_west: float | None = None
    pct_windows_facing_northwest: float | None = None
    foundation_type: str | None = None
    ceiling_type: str | None = None

    primary_heating_system_type: str | None = None
    primary_heating_fuel_type: str | None = None
    primary_cooling_system_type: str | None = None
    primary_water_heater_technology: str | None = None
    primary_water_heater_fuel_type: str | None = None


    def model_dump(self) -> dict[str, Any]:
        """Provide the familiar Pydantic-style serialization method."""
        return asdict(self)


@dataclass
class RBSA_III_BehaviorInput:
    """Cleaned RBSA III occupant, appliance, and energy-use behavior facts."""

    building_id: str
    site_id: str

    qty_occupants: int | None = None
    tvs_total: int | None = None
    tvs_energy_star: int | None = None
    laundry_washer_total: int | None = None
    laundry_washer_energy_star: int | None = None
    laundry_dryer_total: int | None = None
    laundry_dryer_energy_star: int | None = None
    refrigerator_total: int | None = None
    refrigerator_energy_star: int | None = None
    freezer_total: int | None = None
    freezer_energy_star: int | None = None
    stove_oven_total: int | None = None
    air_cleaner_total: int | None = None
    air_cleaner_energy_star: int | None = None
    smart_speaker_total: int | None = None
    dehumidifier_total: int | None = None
    dehumidifier_energy_star: int | None = None
    dishwasher_total: int | None = None
    dishwasher_energy_star: int | None = None

    heating_setpoint: str | float | None = None
    overnight_heating_setpoint: str | float | None = None
    open_windows_when_heating: str | None = None
    block_off_part_of_home_in_heating_season: str | None = None
    percent_of_home_at_different_temp: str | float | None = None
    cooling_setpoint: float | None = None
    overnight_cooling_setpoint: float | None = None
    outdoor_temp_for_cooling: str | float | None = None

    qty_installed_cfl_lamps: int | None = None
    qty_installed_led_lamps: int | None = None
    qty_installed_incandescent_lamps: int | None = None
    qty_installed_halogen_lamps: int | None = None
    qty_installed_linear_fluorescent_lamps: int | None = None
    qty_installed_unknown_lamps: int | None = None
    qty_installed_other_lamps: int | None = None
    total_installed_lamps: int | None = None

    pool_present: str | None = None
    pool_availability_months_per_year: float | None = None
    pool_heat_primary_fuel_type: str | None = None
    pool_heat_primary_fuel_type_other: str | None = None
    pool_solar_assist: str | None = None
    hot_tub_present: str | None = None
    hot_tub_area: str | float | None = None

    number_of_ev_charging_stations: int | None = None

    def model_dump(self) -> dict[str, Any]:
        """Provide the familiar Pydantic-style serialization method."""
        return asdict(self)


SOURCE_MAPPING = {
    "building_id": ("SiteDetail.csv", "Building_ID; falls back to SiteID"),
    "site_id": ("HEMS_RBSA_BASE.csv", "SiteID where RBSA Data Source is RBSA III"),
    #"ee_site_id": ("SITES v9.2.csv", "ee_site_id matched on rbsa_site_id; links to metered parquet RBSA2022_{ee_site_id}"),
    "conditioned_area_ft2": ("Building_Shell_One_Line.csv", "Conditioned_Area"),
    "conditioned_volume_ft3": ("Building_Shell_One_Line.csv", "Conditioned_Volume"),
    "average_height_ft": ("Envelope_Construction.csv", "Average_Height"),
    "conditioned_volume_to_area_ratio_ft": (
        "Building_Shell_One_Line.csv",
        "Conditioned_Volume_to_Area_Ratio",
    ),
    "total_building_levels": ("Envelope_Construction.csv", "Total_Building_Levels"),
    "inferred_total_building_levels": (
        "Calculated",
        "When reported levels are missing: ceil(conditioned_area_ft2 / footprint_area_ft2), minus 1 for >90% Conditioned Basement, minimum 1",
    ),
    "effective_total_building_levels": (
        "Calculated",
        "Reported total_building_levels when available, otherwise inferred_total_building_levels",
    ),
    "footprint_area_ft2": (
        "Envelope_Floor_Foundation.csv",
        "Sum Floor_Area for Basement, Crawlspace, and Slab rows",
    ),
    "footprint_perimeter_ft": (
        "Envelope_Floor_Foundation.csv",
        "Sum Perimeter for Basement, Crawlspace, and Slab rows",
    ),
    "has_conditioned_basement": (
        "Envelope_Construction.csv",
        "Foundation_Type exactly equals >90% Conditioned Basement",
    ),
    "footprint_based_conditioned_area_ft2": (
        "Calculated",
        "footprint_area_ft2 * (effective_total_building_levels + 1 only for >90% Conditioned Basement)",
    ),
    "conditioned_area_difference_ft2": (
        "Calculated",
        "conditioned_area_ft2 - footprint_based_conditioned_area_ft2",
    ),
    "conditioned_area_absolute_difference_ft2": (
        "Calculated",
        "absolute value of conditioned_area_difference_ft2",
    ),
    "conditioned_area_difference_pct": (
        "Calculated",
        "conditioned_area_absolute_difference_ft2 / conditioned_area_ft2 * 100",
    ),
    "footprint_conditioned_area_issue": (
        "Calculated",
        "True when signed difference is greater than +25% or less than -75%",
    ),
    "total_wall_u_value_missing_issue": (
        "Calculated",
        "True when total_wall_u_value_ip is missing",
    ),
    "window_to_wall_ratio_missing_issue": (
        "Calculated",
        "True when window_to_wall_ratio is missing",
    ),
    "metered_data_high_missing_rate": (
        "Calculated",
        "True when the site's 15-min metered parquet has >= 5% missing intervals",
    ),
    "has_data_quality_issue": (
        "Calculated",
        "OR-union of all component data-quality issue flags, including metered_data_high_missing_rate",
    ),
    "window_to_wall_ratio": ("Building_Shell_One_Line.csv", "Window_to_Wall_Ratio"),
    "total_wall_u_value_ip": ("Building_Shell_One_Line.csv", "Total_Wall_U-Value"),
    "total_ceiling_u_value_ip": ("Building_Shell_One_Line.csv", "Total_Ceiling_U-Value"),
    "total_floor_u_value_ip": ("Building_Shell_One_Line.csv", "Total_Floor_U-Value"),
    "window_u_value_ip": ("Building_Shell_One_Line.csv", "Window_U-Value"),
    "ach50": ("Testing_Blowerdoor.csv", "ACH_50"),
    "pct_windows_facing_north": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_North",
    ),
    "pct_windows_facing_northeast": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_NorthEast",
    ),
    "pct_windows_facing_east": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_East",
    ),
    "pct_windows_facing_southeast": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_SouthEast",
    ),
    "pct_windows_facing_south": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_South",
    ),
    "pct_windows_facing_southwest": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_SouthWest",
    ),
    "pct_windows_facing_west": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_West",
    ),
    "pct_windows_facing_northwest": (
        "Envelope_BuildingFenestration.csv",
        "Pct_Windows_Facing_NorthWest",
    ),
    "foundation_type": (
        "Envelope_Construction.csv",
        "Foundation_Type and Foundation_Type_Other",
    ),
    "ceiling_type": (
        "Envelope_Ceiling.csv",
        "Ceiling_Type with largest Ceiling_Area; use Ceiling_Type_Other when applicable",
    ),
    "primary_heating_system_type": (
        "Mechanical_One_Line.csv",
        "Primary_Heating_System_Type",
    ),
    "primary_heating_fuel_type": (
        "Mechanical_One_Line.csv",
        "Primary_Heating_Fuel_Type",
    ),
    "primary_cooling_system_type": (
        "Mechanical_One_Line.csv; Mechanical_HeatingAndCooling.csv",
        "Primary_Cooling_System_Type; No Cooling when all detailed rows say Provides_Cooling=No",
    ),
    "primary_water_heater_technology": (
        "Mechanical_WaterHeater.csv",
        "Technology_Description; prefer Serves_Whole_House=Yes and require agreement",
    ),
    "primary_water_heater_fuel_type": (
        "Mechanical_WaterHeater.csv",
        "Fuel_Type; prefer Serves_Whole_House=Yes and require agreement; infer Natural Gas when missing and technology is Fossil Fuel Non-Condensing",
    ),
}


BEHAVIOR_SOURCE_MAPPING = {
    "building_id": ("SiteDetail.csv", "Building_ID; falls back to SiteID"),
    "site_id": ("HEMS_RBSA_BASE.csv", "SiteID where RBSA Data Source is RBSA III"),
    "qty_occupants": ("Appliance_One_Line.csv", "Qty_Occupants"),
    "tvs_total": ("Appliance_One_Line.csv", "TVs_Total"),
    "tvs_energy_star": ("Appliance_One_Line.csv", "TVs_EnergyStar"),
    "laundry_washer_total": ("Appliance_One_Line.csv", "Laundry_Washer_Total"),
    "laundry_washer_energy_star": (
        "Appliance_One_Line.csv",
        "Laundry_Washer_EnergyStar",
    ),
    "laundry_dryer_total": ("Appliance_One_Line.csv", "Laundry_Dryer_Total"),
    "laundry_dryer_energy_star": (
        "Appliance_One_Line.csv",
        "Laundry_Dryer_EnergyStar",
    ),
    "refrigerator_total": ("Appliance_One_Line.csv", "Refrigerator_Total"),
    "refrigerator_energy_star": (
        "Appliance_One_Line.csv",
        "Refrigerator_EnergyStar",
    ),
    "freezer_total": ("Appliance_One_Line.csv", "Freezer_Total"),
    "freezer_energy_star": ("Appliance_One_Line.csv", "Freezer_EnergyStar"),
    "stove_oven_total": ("Appliance_One_Line.csv", "Stove_Oven_Total"),
    "air_cleaner_total": ("Appliance_One_Line.csv", "Air_Cleaner_Total"),
    "air_cleaner_energy_star": (
        "Appliance_One_Line.csv",
        "Air_Cleaner_EnergyStar",
    ),
    "smart_speaker_total": ("Appliance_One_Line.csv", "Smart_Speaker_Total"),
    "dehumidifier_total": ("Appliance_One_Line.csv", "Dehumidifier_Total"),
    "dehumidifier_energy_star": (
        "Appliance_One_Line.csv",
        "Dehumidifier_EnergyStar",
    ),
    "dishwasher_total": ("Appliance_One_Line.csv", "Dishwasher_Total"),
    "dishwasher_energy_star": (
        "Appliance_One_Line.csv",
        "Dishwasher_EnergyStar",
    ),
    "heating_setpoint": ("SiteInterview_HomeEnergyUse.csv", "Heating_Setpoint"),
    "overnight_heating_setpoint": (
        "SiteInterview_HomeEnergyUse.csv",
        "Overnight_Heating_Setpoint",
    ),
    "open_windows_when_heating": (
        "SiteInterview_HomeEnergyUse.csv",
        "Open_Windows_When_Heating",
    ),
    "block_off_part_of_home_in_heating_season": (
        "SiteInterview_HomeEnergyUse.csv",
        "Block_Off_Part_of_Home_In_Heating_Season",
    ),
    "percent_of_home_at_different_temp": (
        "SiteInterview_HomeEnergyUse.csv",
        "Percent_of_Home_At_Different_Temp",
    ),
    "cooling_setpoint": ("SiteInterview_HomeEnergyUse.csv", "Cooling_Setpoint"),
    "overnight_cooling_setpoint": (
        "SiteInterview_HomeEnergyUse.csv",
        "Overnight_Cooling_Setpoint",
    ),
    "outdoor_temp_for_cooling": (
        "SiteInterview_HomeEnergyUse.csv",
        "Outdoor_Temp_for_Cooling",
    ),
    "qty_installed_cfl_lamps": ("Lighting_One_Line.csv", "Qty_Installed_CFL_Lamps"),
    "qty_installed_led_lamps": ("Lighting_One_Line.csv", "Qty_Installed_LED_Lamps"),
    "qty_installed_incandescent_lamps": (
        "Lighting_One_Line.csv",
        "Qty_Installed_Incandescent_Lamps",
    ),
    "qty_installed_halogen_lamps": (
        "Lighting_One_Line.csv",
        "Qty_Installed_Halogen_Lamps",
    ),
    "qty_installed_linear_fluorescent_lamps": (
        "Lighting_One_Line.csv",
        "Qty_Installed_Linear_Fluor._Lamps",
    ),
    "qty_installed_unknown_lamps": (
        "Lighting_One_Line.csv",
        "Qty_Installed_Unknown_Lamps",
    ),
    "qty_installed_other_lamps": (
        "Lighting_One_Line.csv",
        "Qty_Installed_Other_Lamps",
    ),
    "total_installed_lamps": ("Lighting_One_Line.csv", "Total_Installed_Lamps"),
    "pool_present": ("SiteInterview_HomeEnergyUse.csv", "Pool_Present"),
    "pool_availability_months_per_year": (
        "SiteInterview_HomeEnergyUse.csv",
        "Pool_Availability_Months_Per_Year",
    ),
    "pool_heat_primary_fuel_type": (
        "SiteInterview_HomeEnergyUse.csv",
        "Pool_Heat_Primary_Fuel_Type",
    ),
    "pool_heat_primary_fuel_type_other": (
        "SiteInterview_HomeEnergyUse.csv",
        "Pool_Heat_Primary_Fuel_Type_Other",
    ),
    "pool_solar_assist": ("SiteInterview_HomeEnergyUse.csv", "Pool_Solar_Assist"),
    "hot_tub_present": ("SiteInterview_HomeEnergyUse.csv", "Hot_Tub_Present"),
    "hot_tub_area": ("SiteInterview_HomeEnergyUse.csv", "Hot_Tub_Area"),
    "number_of_ev_charging_stations": (
        "ElectricVehicleChargers.csv",
        "Number_Of_EV_Charging_Stations",
    ),
}

_SENTINELS = {"", "unknown", "not available", "n/a", "na", "none", "null"}


def _clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in _SENTINELS else text


def _clean_float(value: Any) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _clean_float_or_text(value: Any) -> str | float | None:
    text = _clean_text(value)
    if text is None:
        return None
    number = _clean_float(text)
    return number if number is not None else text


def _clean_bounded_float(
    value: Any,
    minimum: float,
    maximum: float,
) -> float | None:
    number = _clean_float(value)
    if number is None or not minimum <= number <= maximum:
        return None
    return number


def _clean_int(value: Any) -> int | None:
    number = _clean_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _clean_percent(value: Any) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    return _clean_float(text.removesuffix("%"))


def _has_conditioned_basement(foundation_type: str | None) -> bool | None:
    if foundation_type is None:
        return None
    return foundation_type.casefold() == ">90% conditioned basement"


def _read_by_site(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str)
    if "SiteID" not in data.columns:
        raise ValueError(f"{path.name} does not contain SiteID")
    return data.drop_duplicates("SiteID").set_index("SiteID")


def _foundation_footprints(path: Path) -> pd.DataFrame:
    """Aggregate the measured basement, crawlspace, and slab footprint sections."""
    data = pd.read_csv(path, dtype=str)
    data = data[data["Floor_Type"].isin(["Basement", "Crawlspace", "Slab"])].copy()
    data["footprint_area_ft2"] = data["Floor_Area"].map(_clean_float)
    data["footprint_perimeter_ft"] = data["Perimeter"].map(_clean_float)
    return data.groupby("SiteID").agg(
        footprint_area_ft2=("footprint_area_ft2", lambda values: values.sum(min_count=1)),
        footprint_perimeter_ft=(
            "footprint_perimeter_ft",
            lambda values: values.sum(min_count=1),
        ),
    )


def _detailed_no_cooling_sites(mechanical_detail: pd.DataFrame) -> set[str]:
    cooling = mechanical_detail["Provides_Cooling"].map(_clean_text)
    detail = mechanical_detail.assign(_provides_cooling=cooling)
    return {
        site_id
        for site_id, rows in detail.groupby("SiteID")
        if rows["_provides_cooling"].notna().all()
        and rows["_provides_cooling"].str.casefold().eq("no").all()
    }


def _primary_water_heaters(path: Path) -> pd.DataFrame:
    """Select agreed primary water-heater technology and fuel for each site."""
    data = pd.read_csv(path, dtype=str).drop_duplicates()
    rows = []
    for site_id, site_rows in data.groupby("SiteID"):
        whole_house = site_rows[
            site_rows["Serves_Whole_House"].fillna("").str.casefold().eq("yes")
        ]
        candidates = whole_house if not whole_house.empty else site_rows

        values = {}
        for output_field, source_column in [
            ("primary_water_heater_technology", "Technology_Description"),
            ("primary_water_heater_fuel_type", "Fuel_Type"),
        ]:
            unique_values = {
                cleaned
                for value in candidates[source_column]
                if (cleaned := _clean_text(value)) is not None
            }
            values[output_field] = unique_values.pop() if len(unique_values) == 1 else None

        if (
            values["primary_water_heater_fuel_type"] is None
        ):
            if values ["primary_water_heater_technology"] == "Fossil Fuel Non-Condensing":
                values["primary_water_heater_fuel_type"] = "Natural Gas"
            else:
                values["primary_water_heater_fuel_type"] = "Electricity"

            

        rows.append({"SiteID": site_id, **values})

    return pd.DataFrame(rows).set_index("SiteID")


def _primary_ceiling_types(path: Path) -> pd.DataFrame:
    """Select the ceiling type covering the largest reported ceiling area."""
    data = pd.read_csv(path, dtype=str)
    data["_area"] = data["Ceiling_Area"].map(_clean_float)
    rows = []

    for site_id, site_rows in data.groupby("SiteID"):
        largest_area = site_rows["_area"].max()
        candidates = (
            site_rows[site_rows["_area"].eq(largest_area)]
            if pd.notna(largest_area)
            else site_rows
        )

        types = set()
        for _, row in candidates.iterrows():
            ceiling_type = _clean_text(row.get("Ceiling_Type"))
            ceiling_other = _clean_text(row.get("Ceiling_Type_Other"))
            if ceiling_type == "Other" and ceiling_other:
                ceiling_type = f"Other: {ceiling_other}"
            if ceiling_type is not None:
                types.add(ceiling_type)

        rows.append(
            {
                "SiteID": site_id,
                "ceiling_type": types.pop() if len(types) == 1 else None,
            }
        )

    return pd.DataFrame(rows).set_index("SiteID")


def build_rbsa_iii_model_inputs(
    rbsa_iii_dir: str | Path,
    parquet_name: list[str,] | None = None,
) -> list[RBSA_III_ModelInput]:
    """Return one model input for every RBSA III site in HEMS_RBSA_BASE.csv.

    Parameters
    ----------
    parquet_name:
        Optional list of parquet file names (stems) to include.
        When provided, each record's ee_site_id is populated from the list and
        metered_data_high_missing_rate is set True for sites absent from the list.
    """
    rbsa_iii_dir = Path(rbsa_iii_dir)
    hems_dir = rbsa_iii_dir / "HEMS"

    base = pd.read_csv(rbsa_iii_dir.parent / "HEMS_RBSA_BASE.csv", dtype=str)
    rbsa_iii_base = base[base["RBSA Data Source"].str.strip().eq("RBSA III")]
    site_ids = rbsa_iii_base["SiteID"].dropna().str.strip().drop_duplicates().tolist()

    site = _read_by_site(hems_dir / "SiteDetail.csv")
    shell = _read_by_site(hems_dir / "Building_Shell_One_Line.csv")
    construction = _read_by_site(hems_dir / "Envelope_Construction.csv")
    footprints = _foundation_footprints(hems_dir / "Envelope_Floor_Foundation.csv")
    mechanical = _read_by_site(hems_dir / "Mechanical_One_Line.csv")
    blowerdoor = _read_by_site(hems_dir / "Testing_Blowerdoor.csv")
    fenestration = _read_by_site(hems_dir / "Envelope_BuildingFenestration.csv")
    water_heaters = _primary_water_heaters(hems_dir / "Mechanical_WaterHeater.csv")
    ceiling_types = _primary_ceiling_types(hems_dir / "Envelope_Ceiling.csv")
    mechanical_detail = pd.read_csv(hems_dir / "Mechanical_HeatingAndCooling.csv", dtype=str)
    no_cooling_sites = _detailed_no_cooling_sites(mechanical_detail)

    records = []
    for site_id in site_ids:
        site_row = site.loc[site_id] if site_id in site.index else pd.Series(dtype=object)
        shell_row = shell.loc[site_id] if site_id in shell.index else pd.Series(dtype=object)
        construction_row = (
            construction.loc[site_id] if site_id in construction.index else pd.Series(dtype=object)
        )
        footprint_row = (
            footprints.loc[site_id] if site_id in footprints.index else pd.Series(dtype=object)
        )
        mechanical_row = (
            mechanical.loc[site_id] if site_id in mechanical.index else pd.Series(dtype=object)
        )
        blowerdoor_row = (
            blowerdoor.loc[site_id] if site_id in blowerdoor.index else pd.Series(dtype=object)
        )
        fenestration_row = (
            fenestration.loc[site_id]
            if site_id in fenestration.index
            else pd.Series(dtype=object)
        )
        water_heater_row = (
            water_heaters.loc[site_id]
            if site_id in water_heaters.index
            else pd.Series(dtype=object)
        )
        ceiling_row = (
            ceiling_types.loc[site_id]
            if site_id in ceiling_types.index
            else pd.Series(dtype=object)
        )

        foundation_type = _clean_text(construction_row.get("Foundation_Type"))
        foundation_other = _clean_text(construction_row.get("Foundation_Type_Other"))
        if foundation_type == "Other" and foundation_other:
            foundation_type = f"Other: {foundation_other}"
        conditioned_area = _clean_float(shell_row.get("Conditioned_Area"))
        total_building_levels = _clean_int(construction_row.get("Total_Building_Levels"))
        footprint_area = _clean_float(footprint_row.get("footprint_area_ft2"))
        has_conditioned_basement = _has_conditioned_basement(foundation_type)

        inferred_total_building_levels = None
        if (
            total_building_levels is None
            and conditioned_area is not None
            and footprint_area is not None
            and footprint_area > 0
        ):
            inferred_total_building_levels = math.ceil(conditioned_area / footprint_area)
            if has_conditioned_basement:
                inferred_total_building_levels -= 1
            inferred_total_building_levels = max(1, inferred_total_building_levels)

        effective_total_building_levels = (
            total_building_levels
            if total_building_levels is not None
            else inferred_total_building_levels
        )

        footprint_based_conditioned_area = None
        if footprint_area is not None and effective_total_building_levels is not None:
            basement_levels = 1 if has_conditioned_basement else 0
            footprint_based_conditioned_area = footprint_area * (
                effective_total_building_levels + basement_levels
            )

        conditioned_area_difference = None
        conditioned_area_absolute_difference = None
        conditioned_area_difference_pct = None
        footprint_conditioned_area_issue = None
        if conditioned_area is not None and footprint_based_conditioned_area is not None:
            conditioned_area_difference = conditioned_area - footprint_based_conditioned_area
            conditioned_area_absolute_difference = abs(conditioned_area_difference)
            if conditioned_area > 0:
                conditioned_area_difference_pct = (
                    conditioned_area_absolute_difference / conditioned_area * 100
                )
                signed_difference_pct = conditioned_area_difference / conditioned_area * 100
                footprint_conditioned_area_issue = (
                    signed_difference_pct > 25 or signed_difference_pct < -75
                )

        primary_cooling = _clean_text(mechanical_row.get("Primary_Cooling_System_Type"))
        if primary_cooling is None and site_id in no_cooling_sites:
            primary_cooling = "No Cooling"

        window_to_wall_ratio = _clean_float(shell_row.get("Window_to_Wall_Ratio"))
        total_wall_u_value_ip = _clean_float(shell_row.get("Total_Wall_U-Value"))
        total_wall_u_value_missing_issue = total_wall_u_value_ip is None
        window_to_wall_ratio_missing_issue = window_to_wall_ratio is None
        #parquet_stem = parquet_name_map.get(site_id) if parquet_name_map is not None else None
        metered_data_high_missing_rate = (
            parquet_name is not None and site_id not in parquet_name
        )
        has_data_quality_issue = any(
            (
                footprint_conditioned_area_issue is True,
                total_wall_u_value_missing_issue,
                window_to_wall_ratio_missing_issue,
                metered_data_high_missing_rate,
            )
        )

        records.append(
            RBSA_III_ModelInput(
                building_id=_clean_text(site_row.get("Building_ID")) or site_id,
                site_id=site_id,
                conditioned_area_ft2=conditioned_area,
                conditioned_volume_ft3=_clean_float(shell_row.get("Conditioned_Volume")),
                average_height_ft=_clean_float(construction_row.get("Average_Height")),
                conditioned_volume_to_area_ratio_ft=_clean_float(
                    shell_row.get("Conditioned_Volume_to_Area_Ratio")
                ),
                total_building_levels=total_building_levels,
                inferred_total_building_levels=inferred_total_building_levels,
                effective_total_building_levels=effective_total_building_levels,
                footprint_area_ft2=footprint_area,
                footprint_perimeter_ft=_clean_float(
                    footprint_row.get("footprint_perimeter_ft")
                ),
                has_conditioned_basement=has_conditioned_basement,
                footprint_based_conditioned_area_ft2=footprint_based_conditioned_area,
                conditioned_area_difference_ft2=conditioned_area_difference,
                conditioned_area_absolute_difference_ft2=conditioned_area_absolute_difference,
                conditioned_area_difference_pct=conditioned_area_difference_pct,
                footprint_conditioned_area_issue=footprint_conditioned_area_issue,
                total_wall_u_value_missing_issue=total_wall_u_value_missing_issue,
                window_to_wall_ratio_missing_issue=window_to_wall_ratio_missing_issue,
                metered_data_high_missing_rate=metered_data_high_missing_rate,
                has_data_quality_issue=has_data_quality_issue,
                window_to_wall_ratio=window_to_wall_ratio,
                total_wall_u_value_ip=total_wall_u_value_ip,
                total_ceiling_u_value_ip=_clean_float(
                    shell_row.get("Total_Ceiling_U-Value")
                ),
                total_floor_u_value_ip=_clean_float(shell_row.get("Total_Floor_U-Value")),
                window_u_value_ip=_clean_float(shell_row.get("Window_U-Value")),
                ach50=_clean_float(blowerdoor_row.get("ACH_50")),
                pct_windows_facing_north=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_North")
                ),
                pct_windows_facing_northeast=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_NorthEast")
                ),
                pct_windows_facing_east=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_East")
                ),
                pct_windows_facing_southeast=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_SouthEast")
                ),
                pct_windows_facing_south=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_South")
                ),
                pct_windows_facing_southwest=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_SouthWest")
                ),
                pct_windows_facing_west=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_West")
                ),
                pct_windows_facing_northwest=_clean_percent(
                    fenestration_row.get("Pct_Windows_Facing_NorthWest")
                ),
                foundation_type=foundation_type,
                ceiling_type=_clean_text(ceiling_row.get("ceiling_type")),
                primary_heating_system_type=_clean_text(
                    mechanical_row.get("Primary_Heating_System_Type")
                ),
                primary_heating_fuel_type=_clean_text(
                    mechanical_row.get("Primary_Heating_Fuel_Type")
                ),
                primary_cooling_system_type=primary_cooling,
                primary_water_heater_technology=_clean_text(
                    water_heater_row.get("primary_water_heater_technology")
                ),
                primary_water_heater_fuel_type=_clean_text(
                    water_heater_row.get("primary_water_heater_fuel_type")
                ),
            )
        )

    return records


def build_rbsa_iii_behavior_inputs(
    rbsa_iii_dir: str | Path,
) -> list[RBSA_III_BehaviorInput]:
    """Return one behavior input for every RBSA III site in HEMS_RBSA_BASE.csv."""
    rbsa_iii_dir = Path(rbsa_iii_dir)
    hems_dir = rbsa_iii_dir / "HEMS"

    base = pd.read_csv(rbsa_iii_dir.parent / "HEMS_RBSA_BASE.csv", dtype=str)
    rbsa_iii_base = base[base["RBSA Data Source"].str.strip().eq("RBSA III")]
    site_ids = rbsa_iii_base["SiteID"].dropna().str.strip().drop_duplicates().tolist()

    site = _read_by_site(hems_dir / "SiteDetail.csv")
    appliances = _read_by_site(hems_dir / "Appliance_One_Line.csv")
    home_energy_use = _read_by_site(hems_dir / "SiteInterview_HomeEnergyUse.csv")
    lighting = _read_by_site(hems_dir / "Lighting_One_Line.csv")
    ev_chargers = _read_by_site(hems_dir / "ElectricVehicleChargers.csv")

    records = []
    for site_id in site_ids:
        site_row = site.loc[site_id] if site_id in site.index else pd.Series(dtype=object)
        appliance_row = (
            appliances.loc[site_id]
            if site_id in appliances.index
            else pd.Series(dtype=object)
        )
        home_energy_use_row = (
            home_energy_use.loc[site_id]
            if site_id in home_energy_use.index
            else pd.Series(dtype=object)
        )
        lighting_row = (
            lighting.loc[site_id] if site_id in lighting.index else pd.Series(dtype=object)
        )
        ev_charger_row = (
            ev_chargers.loc[site_id]
            if site_id in ev_chargers.index
            else pd.Series(dtype=object)
        )

        records.append(
            RBSA_III_BehaviorInput(
                building_id=_clean_text(site_row.get("Building_ID")) or site_id,
                site_id=site_id,
                qty_occupants=_clean_int(appliance_row.get("Qty_Occupants")),
                tvs_total=_clean_int(appliance_row.get("TVs_Total")),
                tvs_energy_star=_clean_int(appliance_row.get("TVs_EnergyStar")),
                laundry_washer_total=_clean_int(
                    appliance_row.get("Laundry_Washer_Total")
                ),
                laundry_washer_energy_star=_clean_int(
                    appliance_row.get("Laundry_Washer_EnergyStar")
                ),
                laundry_dryer_total=_clean_int(appliance_row.get("Laundry_Dryer_Total")),
                laundry_dryer_energy_star=_clean_int(
                    appliance_row.get("Laundry_Dryer_EnergyStar")
                ),
                refrigerator_total=_clean_int(appliance_row.get("Refrigerator_Total")),
                refrigerator_energy_star=_clean_int(
                    appliance_row.get("Refrigerator_EnergyStar")
                ),
                freezer_total=_clean_int(appliance_row.get("Freezer_Total")),
                freezer_energy_star=_clean_int(appliance_row.get("Freezer_EnergyStar")),
                stove_oven_total=_clean_int(appliance_row.get("Stove_Oven_Total")),
                air_cleaner_total=_clean_int(appliance_row.get("Air_Cleaner_Total")),
                air_cleaner_energy_star=_clean_int(
                    appliance_row.get("Air_Cleaner_EnergyStar")
                ),
                smart_speaker_total=_clean_int(appliance_row.get("Smart_Speaker_Total")),
                dehumidifier_total=_clean_int(appliance_row.get("Dehumidifier_Total")),
                dehumidifier_energy_star=_clean_int(
                    appliance_row.get("Dehumidifier_EnergyStar")
                ),
                dishwasher_total=_clean_int(appliance_row.get("Dishwasher_Total")),
                dishwasher_energy_star=_clean_int(
                    appliance_row.get("Dishwasher_EnergyStar")
                ),
                heating_setpoint=_clean_float_or_text(
                    home_energy_use_row.get("Heating_Setpoint")
                ),
                overnight_heating_setpoint=_clean_float_or_text(
                    home_energy_use_row.get("Overnight_Heating_Setpoint")
                ),
                open_windows_when_heating=_clean_text(
                    home_energy_use_row.get("Open_Windows_When_Heating")
                ),
                block_off_part_of_home_in_heating_season=_clean_text(
                    home_energy_use_row.get(
                        "Block_Off_Part_of_Home_In_Heating_Season"
                    )
                ),
                percent_of_home_at_different_temp=_clean_float_or_text(
                    home_energy_use_row.get("Percent_of_Home_At_Different_Temp")
                ),
                cooling_setpoint=_clean_float(
                    home_energy_use_row.get("Cooling_Setpoint")
                ),
                overnight_cooling_setpoint=_clean_float(
                    home_energy_use_row.get("Overnight_Cooling_Setpoint")
                ),
                outdoor_temp_for_cooling=_clean_float_or_text(
                    home_energy_use_row.get("Outdoor_Temp_for_Cooling")
                ),
                qty_installed_cfl_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_CFL_Lamps")
                ),
                qty_installed_led_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_LED_Lamps")
                ),
                qty_installed_incandescent_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_Incandescent_Lamps")
                ),
                qty_installed_halogen_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_Halogen_Lamps")
                ),
                qty_installed_linear_fluorescent_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_Linear_Fluor._Lamps")
                ),
                qty_installed_unknown_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_Unknown_Lamps")
                ),
                qty_installed_other_lamps=_clean_int(
                    lighting_row.get("Qty_Installed_Other_Lamps")
                ),
                total_installed_lamps=_clean_int(
                    lighting_row.get("Total_Installed_Lamps")
                ),
                pool_present=_clean_text(home_energy_use_row.get("Pool_Present")),
                pool_availability_months_per_year=_clean_bounded_float(
                    home_energy_use_row.get("Pool_Availability_Months_Per_Year"),
                    minimum=0,
                    maximum=12,
                ),
                pool_heat_primary_fuel_type=_clean_text(
                    home_energy_use_row.get("Pool_Heat_Primary_Fuel_Type")
                ),
                pool_heat_primary_fuel_type_other=_clean_text(
                    home_energy_use_row.get("Pool_Heat_Primary_Fuel_Type_Other")
                ),
                pool_solar_assist=_clean_text(
                    home_energy_use_row.get("Pool_Solar_Assist")
                ),
                hot_tub_present=_clean_text(
                    home_energy_use_row.get("Hot_Tub_Present")
                ),
                hot_tub_area=_clean_float_or_text(
                    home_energy_use_row.get("Hot_Tub_Area")
                ),
                number_of_ev_charging_stations=_clean_int(
                    ev_charger_row.get("Number_Of_EV_Charging_Stations")
                ),
            )
        )

    return records


def export_rbsa_iii_model_inputs(
    rbsa_iii_dir: str | Path,
    parquet_name: list[str] | None = None,
) -> dict[str, Path | int]:
    """Save all RBSA III model inputs as a JSON array and companion CSV."""
    rbsa_iii_dir = Path(rbsa_iii_dir)
    records = build_rbsa_iii_model_inputs(rbsa_iii_dir, parquet_name=parquet_name)
    output_json = rbsa_iii_dir / "RBSA_III_model_inputs.json"
    output_csv = rbsa_iii_dir / "RBSA_III_model_inputs.csv"
    mapping_csv = rbsa_iii_dir / "RBSA_III_model_input_source_mapping.csv"

    dumped = [record.model_dump() for record in records]
    output_json.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
    pd.DataFrame(dumped).to_csv(output_csv, index=False)
    pd.DataFrame(
        [
            {"model_field": field, "source_file": source[0], "source_column": source[1]}
            for field, source in SOURCE_MAPPING.items()
        ]
    ).to_csv(mapping_csv, index=False)

    return {
        "records": len(records),
        "json": output_json,
        "csv": output_csv,
        "mapping_csv": mapping_csv,
    }


def export_rbsa_iii_behavior_inputs(
    rbsa_iii_dir: str | Path,
) -> dict[str, Path | int]:
    """Save all RBSA III behavior inputs as a JSON array and companion CSV."""
    rbsa_iii_dir = Path(rbsa_iii_dir)
    records = build_rbsa_iii_behavior_inputs(rbsa_iii_dir)
    output_json = rbsa_iii_dir / "RBSA_III_behavior_inputs.json"
    output_csv = rbsa_iii_dir / "RBSA_III_behavior_inputs.csv"
    mapping_csv = rbsa_iii_dir / "RBSA_III_behavior_input_source_mapping.csv"

    dumped = [record.model_dump() for record in records]
    output_json.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
    pd.DataFrame(dumped).to_csv(output_csv, index=False)
    pd.DataFrame(
        [
            {"behavior_field": field, "source_file": source[0], "source_column": source[1]}
            for field, source in BEHAVIOR_SOURCE_MAPPING.items()
        ]
    ).to_csv(mapping_csv, index=False)

    return {
        "records": len(records),
        "json": output_json,
        "csv": output_csv,
        "mapping_csv": mapping_csv,
    }
