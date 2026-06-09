"""Build one model-input record for every RBSA III site in HEMS_RBSA_BASE."""

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

    conditioned_area_ft2: float | None = None
    conditioned_volume_ft3: float | None = None
    average_height_ft: float | None = None
    conditioned_volume_to_area_ratio_ft: float | None = None
    total_building_levels: int | None = None
    footprint_area_ft2: float | None = None
    footprint_perimeter_ft: float | None = None

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


SOURCE_MAPPING = {
    "building_id": ("SiteDetail.csv", "Building_ID; falls back to SiteID"),
    "site_id": ("HEMS_RBSA_BASE.csv", "SiteID where RBSA Data Source is RBSA III"),
    "conditioned_area_ft2": ("Building_Shell_One_Line.csv", "Conditioned_Area"),
    "conditioned_volume_ft3": ("Building_Shell_One_Line.csv", "Conditioned_Volume"),
    "average_height_ft": ("Envelope_Construction.csv", "Average_Height"),
    "conditioned_volume_to_area_ratio_ft": (
        "Building_Shell_One_Line.csv",
        "Conditioned_Volume_to_Area_Ratio",
    ),
    "total_building_levels": ("Envelope_Construction.csv", "Total_Building_Levels"),
    "footprint_area_ft2": (
        "Envelope_Floor_Foundation.csv",
        "Sum Floor_Area for Basement, Crawlspace, and Slab rows",
    ),
    "footprint_perimeter_ft": (
        "Envelope_Floor_Foundation.csv",
        "Sum Perimeter for Basement, Crawlspace, and Slab rows",
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
        "Fuel_Type; prefer Serves_Whole_House=Yes and require agreement",
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


def build_rbsa_iii_model_inputs(rbsa_iii_dir: str | Path) -> list[RBSA_III_ModelInput]:
    """Return one model input for every RBSA III site in HEMS_RBSA_BASE.csv."""
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

        primary_cooling = _clean_text(mechanical_row.get("Primary_Cooling_System_Type"))
        if primary_cooling is None and site_id in no_cooling_sites:
            primary_cooling = "No Cooling"

        records.append(
            RBSA_III_ModelInput(
                building_id=_clean_text(site_row.get("Building_ID")) or site_id,
                site_id=site_id,
                conditioned_area_ft2=_clean_float(shell_row.get("Conditioned_Area")),
                conditioned_volume_ft3=_clean_float(shell_row.get("Conditioned_Volume")),
                average_height_ft=_clean_float(construction_row.get("Average_Height")),
                conditioned_volume_to_area_ratio_ft=_clean_float(
                    shell_row.get("Conditioned_Volume_to_Area_Ratio")
                ),
                total_building_levels=_clean_int(construction_row.get("Total_Building_Levels")),
                footprint_area_ft2=_clean_float(footprint_row.get("footprint_area_ft2")),
                footprint_perimeter_ft=_clean_float(
                    footprint_row.get("footprint_perimeter_ft")
                ),
                window_to_wall_ratio=_clean_float(shell_row.get("Window_to_Wall_Ratio")),
                total_wall_u_value_ip=_clean_float(shell_row.get("Total_Wall_U-Value")),
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


def export_rbsa_iii_model_inputs(rbsa_iii_dir: str | Path) -> dict[str, Path | int]:
    """Save all RBSA III model inputs as a JSON array and companion CSV."""
    rbsa_iii_dir = Path(rbsa_iii_dir)
    records = build_rbsa_iii_model_inputs(rbsa_iii_dir)

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
