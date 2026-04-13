"""
build_amy_epw.py
----------------
Build an Actual Meteorological Year (AMY) EPW file for a given NOAA ISD
station ID and year.

Pipeline
--------
1. Fetch station metadata (lat/lon/elev/timezone) from ISD history file
2. Download ISD global-hourly CSV for that station + year
3. Parse and resample to a clean 8760-row hourly DataFrame
4. Fill met-field gaps:
   - Short gaps (<=3 h): linear interpolation
   - Longer gaps: hourly climatological mean (same hour-of-day × month)
5. Fetch actual-year solar data (DNI/DHI/GHI) via NSRDB PSM4 aggregated API
6. Derive remaining EPW fields (horizontal IR, etc.)
7. Write EPW file via ladybug

Requirements
------------
    pip install requests pandas numpy pvlib ladybug-core

Usage
-----
    python build_amy_epw.py \
        --station  726300-14733 \
        --year     2023 \
        --api_key  YOUR_NREL_API_KEY \
        --email    you@example.com \
        --out      portland_2023.epw
"""

import argparse
import io
import gzip
import requests
import numpy as np
import pandas as pd
import pvlib
from ladybug.epw import EPW
from ladybug.dt import DateTime
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.header import Header
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype import (
    DryBulbTemperature, DewPointTemperature, RelativeHumidity,
    AtmosphericStationPressure, HorizontalInfraredRadiationIntensity,
    GlobalHorizontalRadiation, DirectNormalRadiation, DiffuseHorizontalRadiation,
    WindDirection, WindSpeed, TotalSkyCover, OpaqueSkyCover,
    Visibility, CeilingHeight, PrecipitableWater,
)

# ---------------------------------------------------------------------------
# 1.  ISD STATION METADATA
# ---------------------------------------------------------------------------

ISD_HISTORY_URL = (
    "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
)

def fetch_station_metadata(station_id: str) -> dict:
    """
    Download the ISD history file and return metadata for the given station.

    station_id format: "USAF-WBAN"  e.g. "726300-14733"
    Returns dict with keys: lat, lon, elev_m, name, state, country, tz_offset
    """
    print(f"  Fetching ISD station metadata for {station_id} ...")
    usaf, wban = station_id.split("-")

    resp = requests.get(ISD_HISTORY_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)

    # Column names vary slightly — normalise
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    row = df[(df["USAF"] == usaf) & (df["WBAN"] == wban)]
    if row.empty:
        raise ValueError(
            f"Station {station_id} not found in ISD history. "
            "Check USAF and WBAN portions."
        )
    row = row.iloc[0]

    lat  = float(row["LAT"])
    lon  = float(row["LON"])
    elev = float(row["ELEV(M)"]) if "ELEV(M)" in row.index else 0.0
    name = row.get("STATION_NAME", station_id)
    state   = row.get("STATE", "")
    country = row.get("CTRY", "")

    # Rough UTC offset from longitude (ISD has no TZ field; refine if needed)
    tz_offset = round(lon / 15)

    print(f"    → {name} ({lat:.4f}, {lon:.4f}), elev={elev} m, UTC{tz_offset:+d}")
    return dict(
        lat=lat, lon=lon, elev_m=elev,
        name=name, state=state, country=country,
        tz_offset=tz_offset,
        usaf=usaf, wban=wban,
    )


# ---------------------------------------------------------------------------
# 2.  ISD GLOBAL-HOURLY DATA
# ---------------------------------------------------------------------------

ISD_DATA_URL = (
    "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv"
)

# Scaling factors and missing-value sentinels from ISD format documentation
ISD_SENTINEL = {
    "TMP": 9999,  "DEW": 9999,  "SLP": 99999,
    "WND": 9999,  "VIS": 999999, "CIG": 99999,
    "AA1": 9999,
}

def fetch_isd_hourly(station_meta: dict, year: int) -> pd.DataFrame:
    """
    Download ISD global-hourly CSV for one station + year.
    Returns a raw DataFrame indexed by UTC datetime.
    """
    usaf = station_meta["usaf"]
    wban = station_meta["wban"]
    url  = ISD_DATA_URL.format(year=year, usaf=usaf, wban=wban.zfill(5))

    print(f"  Downloading ISD data from:\n    {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        low_memory=False,
        parse_dates=["DATE"],
    )
    df = df.rename(columns={"DATE": "utc_time"})
    df = df.set_index("utc_time").sort_index()
    print(f"    → {len(df)} raw observations")
    return df


def parse_isd_field(series: pd.Series, scale: float, sentinel) -> pd.Series:
    """Extract numeric value from ISD comma-delimited field, apply scale."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        first = str(val).split(",")[0]
        try:
            v = float(first)
            return np.nan if v == sentinel else v * scale
        except ValueError:
            return np.nan
    return series.apply(_parse)


def process_isd_to_hourly(df_raw: pd.DataFrame, year: int,
                           tz_offset: int) -> pd.DataFrame:
    """
    Parse ISD raw fields → clean hourly DataFrame in LOCAL time,
    8760 rows (or 8784 for leap year), one row per hour.

    Columns returned:
        temp_C, dewp_C, rh_pct, pressure_Pa, wind_dir_deg, wind_speed_ms,
        total_sky_cover, ceiling_m, visibility_m, precip_mm
    """
    out = pd.DataFrame()

    # Temperature  [scaled ×0.1, sentinel 9999]
    if "TMP" in df_raw.columns:
        out["temp_C"] = parse_isd_field(df_raw["TMP"], 0.1, 9999)

    # Dew point  [scaled ×0.1, sentinel 9999]
    if "DEW" in df_raw.columns:
        out["dewp_C"] = parse_isd_field(df_raw["DEW"], 0.1, 9999)

    # Sea-level pressure  [scaled ×0.1 Pa→hPa, convert to Pa]
    if "SLP" in df_raw.columns:
        slp_hpa = parse_isd_field(df_raw["SLP"], 0.1, 99999)
        out["pressure_Pa"] = slp_hpa * 100.0  # hPa → Pa

    # Wind: "WND" field = "dir,dir_quality,type,speed,speed_quality"
    if "WND" in df_raw.columns:
        def _wind(val):
            if pd.isna(val):
                return np.nan, np.nan
            parts = str(val).split(",")
            try:
                wd = float(parts[0])
                ws = float(parts[3]) * 0.1  # tenths of m/s
                wd = np.nan if wd == 999 else wd
                ws = np.nan if ws * 10 == 9999 else ws
                return wd, ws
            except (IndexError, ValueError):
                return np.nan, np.nan

        wnd_parsed = df_raw["WND"].apply(_wind)
        out["wind_dir_deg"]  = wnd_parsed.apply(lambda x: x[0])
        out["wind_speed_ms"] = wnd_parsed.apply(lambda x: x[1])

    # Ceiling height  [scaled ×1, sentinel 99999]
    if "CIG" in df_raw.columns:
        out["ceiling_m"] = parse_isd_field(df_raw["CIG"], 1.0, 99999)

    # Visibility  [scaled ×1, sentinel 999999]
    if "VIS" in df_raw.columns:
        out["visibility_m"] = parse_isd_field(df_raw["VIS"], 1.0, 999999)

    # Sky cover from cloud coverage field "GD1" if present
    # Encode oktas (0-8) → tenths (0-10) roughly
    if "GD1" in df_raw.columns:
        def _sky(val):
            if pd.isna(val):
                return np.nan
            parts = str(val).split(",")
            try:
                oktas = float(parts[1])
                return np.nan if oktas >= 9 else round(oktas * 10 / 8)
            except (IndexError, ValueError):
                return np.nan
        out["total_sky_cover"] = df_raw["GD1"].apply(_sky)
    else:
        out["total_sky_cover"] = np.nan

    # Precipitation (AA1 field: "period_hours,depth_mm,condition,quality")
    if "AA1" in df_raw.columns:
        out["precip_mm"] = parse_isd_field(df_raw["AA1"], 0.1, 9999)
    else:
        out["precip_mm"] = 0.0

    # ---- Resample to hourly (pick last valid obs in each UTC hour) ----------
    out.index = df_raw.index  # UTC timestamps

    # Keep only the target year
    out = out[out.index.year == year]

    # Resample: for each hour take the last non-NaN observation
    out_hourly = out.resample("1h").last()

    # Build complete 8760-row index in UTC
    freq = "1h"
    full_utc = pd.date_range(
        start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:00",
        freq=freq, tz="UTC"
    )
    out_hourly.index = pd.DatetimeIndex(out_hourly.index).tz_localize("UTC")
    out_hourly = out_hourly.reindex(full_utc)

    # Convert to local standard time (shift by tz_offset hours)
    out_hourly.index = out_hourly.index.shift(tz_offset, freq="1h")
    out_hourly.index = out_hourly.index.tz_localize(None)  # strip tz for EPW

    # Derived: relative humidity from temp + dewpoint
    if "temp_C" in out_hourly and "dewp_C" in out_hourly:
        T  = out_hourly["temp_C"]
        Td = out_hourly["dewp_C"]
        # Magnus formula approximation
        rh = 100 * np.exp((17.625 * Td) / (243.04 + Td)) / \
                   np.exp((17.625 * T)  / (243.04 + T))
        out_hourly["rh_pct"] = rh.clip(0, 100)

    print(f"    → Resampled to {len(out_hourly)} hourly rows (local time)")
    return out_hourly


# ---------------------------------------------------------------------------
# 3.  GAP FILLING
# ---------------------------------------------------------------------------

def fill_gaps(df: pd.DataFrame, short_gap_hours: int = 3) -> pd.DataFrame:
    """
    Two-pass gap filling per column:
      Pass 1 – linear interpolation for gaps <= short_gap_hours
      Pass 2 – climatological mean (hour-of-day × month) for remaining gaps
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue

        n_missing_before = df[col].isna().sum()

        # Pass 1: short-gap linear interpolation
        df[col] = df[col].interpolate(
            method="linear",
            limit=short_gap_hours,
            limit_direction="both"
        )

        # Pass 2: climatological mean for remaining NaNs
        still_missing = df[col].isna()
        if still_missing.any():
            clim = (
                df[col]
                .groupby([df.index.month, df.index.hour])
                .transform("mean")
            )
            df.loc[still_missing, col] = clim[still_missing]

        n_missing_after = df[col].isna().sum()
        if n_missing_before > 0:
            print(
                f"    {col}: filled {n_missing_before - n_missing_after} / "
                f"{n_missing_before} missing values "
                f"({n_missing_after} remain)"
            )

    return df


# ---------------------------------------------------------------------------
# 4.  NSRDB PSM4 SOLAR DATA
# ---------------------------------------------------------------------------

def fetch_solar_psm4(lat: float, lon: float, year: int,
                     api_key: str, email: str) -> pd.DataFrame:
    """
    Retrieve actual-year hourly DNI / DHI / GHI from NSRDB PSM4.
    Returns DataFrame with columns: ghi, dni, dhi (W/m²), local time index.
    """
    print(f"  Fetching NSRDB PSM4 solar for {lat:.4f},{lon:.4f}, year={year} ...")
    solar_df, meta = pvlib.iotools.get_nsrdb_psm4_aggregated(
        latitude=lat,
        longitude=lon,
        api_key=api_key,
        email=email,
        year=year,
        time_step=60,
        parameters=("ghi", "dni", "dhi", "surface_pressure",
                     "air_temperature", "wind_speed"),
        map_variables=True,
    )
    print(f"    → {len(solar_df)} rows, timezone offset: {meta.get('Time Zone', '?')}")
    solar_df = solar_df[["ghi", "dni", "dhi"]].copy()
    solar_df.index = solar_df.index.tz_localize(None)  # strip tz
    return solar_df


# ---------------------------------------------------------------------------
# 5.  DERIVED EPW FIELDS
# ---------------------------------------------------------------------------

def calc_horizontal_ir(temp_C: pd.Series,
                        dewp_C: pd.Series,
                        total_sky_cover: pd.Series) -> pd.Series:
    """
    Estimate horizontal infrared radiation (W/m²) using the
    Berdahl & Martin (1984) formula adapted for EPW:

        IR = sigma * T_sky^4
        T_sky = T_air * (0.711 + 0.56*(Td/100) + 0.73*(N/10)^2)^0.25

    where N is sky cover in tenths (0-10).
    """
    sigma = 5.6697e-8  # Stefan-Boltzmann
    T_K   = temp_C + 273.15
    Td_K  = dewp_C + 273.15
    N     = (total_sky_cover.fillna(5) / 10.0).clip(0, 1)

    emissivity = (0.711 + 0.56 * (Td_K / 273.15 - 1) + 0.73 * N**2).clip(0, 1)
    IR = emissivity * sigma * T_K**4
    return IR.clip(0)


# ---------------------------------------------------------------------------
# 6.  EPW WRITER (via ladybug)
# ---------------------------------------------------------------------------

def write_epw(met: pd.DataFrame, solar: pd.DataFrame,
              station_meta: dict, year: int, out_path: str):
    """
    Merge met + solar DataFrames and write an EPW file using ladybug.

    Both DataFrames must have a full 8760-row DatetimeIndex (no tz).
    """
    print(f"  Writing EPW → {out_path}")

    # Align on a common 8760-row index
    epw_index = pd.date_range(
        start=f"{year}-01-01 01:00",   # EPW hours are end-of-hour
        periods=8760,
        freq="1h"
    )

    def _align(df):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
        # If lengths differ, reindex
        if len(df) != 8760:
            df = df.reindex(epw_index, method="nearest", tolerance="1h")
        else:
            df.index = epw_index
        return df

    met   = _align(met)
    solar = _align(solar)

    # ---- Build EPW object --------------------------------------------------
    epw = EPW()

    lat        = station_meta["lat"]
    lon        = station_meta["lon"]
    tz_offset  = station_meta["tz_offset"]
    elev       = station_meta["elev_m"]
    loc_name   = station_meta["name"]
    state      = station_meta.get("state", "")
    country    = station_meta.get("country", "")

    # Location header
    epw.location.latitude      = lat
    epw.location.longitude     = lon
    epw.location.time_zone     = tz_offset
    epw.location.elevation     = elev
    epw.location.city          = loc_name
    epw.location.state         = state
    epw.location.country       = country
    epw.location.source        = "NOAA-ISD + NSRDB-PSM4 AMY"
    epw.location.station_id    = f"{station_meta['usaf']}-{station_meta['wban']}"

    ap = AnalysisPeriod()  # full year

    def _make_collection(values, data_type, unit):
        header = Header(data_type, unit, ap)
        return HourlyContinuousCollection(header, list(values))

    def _get(df, col, default):
        if col in df.columns:
            vals = df[col].fillna(default).values
        else:
            vals = np.full(8760, default)
        return vals

    temp_C  = _get(met, "temp_C",        20.0)
    dewp_C  = _get(met, "dewp_C",        10.0)
    rh      = _get(met, "rh_pct",        50.0)
    press   = _get(met, "pressure_Pa",   101325.0)
    wd      = _get(met, "wind_dir_deg",  0.0)
    ws      = _get(met, "wind_speed_ms", 0.0)
    sky     = _get(met, "total_sky_cover", 5.0)
    ceil    = _get(met, "ceiling_m",     9999.0)
    vis     = _get(met, "visibility_m",  9999.0)

    ghi = _get(solar, "ghi", 0.0).clip(0)
    dni = _get(solar, "dni", 0.0).clip(0)
    dhi = _get(solar, "dhi", 0.0).clip(0)

    ir = calc_horizontal_ir(
        pd.Series(temp_C), pd.Series(dewp_C), pd.Series(sky)
    ).values

    epw.dry_bulb_temperature              = _make_collection(temp_C,  DryBulbTemperature(),                   "C")
    epw.dew_point_temperature             = _make_collection(dewp_C,  DewPointTemperature(),                  "C")
    epw.relative_humidity                 = _make_collection(rh,      RelativeHumidity(),                     "%")
    epw.atmospheric_station_pressure      = _make_collection(press,   AtmosphericStationPressure(),            "Pa")
    epw.horizontal_infrared_radiation_intensity = _make_collection(ir, HorizontalInfraredRadiationIntensity(), "W/m2")
    epw.global_horizontal_radiation       = _make_collection(ghi,     GlobalHorizontalRadiation(),            "Wh/m2")
    epw.direct_normal_radiation           = _make_collection(dni,     DirectNormalRadiation(),                "Wh/m2")
    epw.diffuse_horizontal_radiation      = _make_collection(dhi,     DiffuseHorizontalRadiation(),           "Wh/m2")
    epw.wind_direction                    = _make_collection(wd,      WindDirection(),                        "deg")
    epw.wind_speed                        = _make_collection(ws,      WindSpeed(),                            "m/s")
    epw.total_sky_cover                   = _make_collection(sky.clip(0, 10), TotalSkyCover(),                "tenths")
    epw.opaque_sky_cover                  = _make_collection(sky.clip(0, 10), OpaqueSkyCover(),               "tenths")
    epw.visibility                        = _make_collection(vis / 1000.0, Visibility(),                     "km")
    epw.ceiling_height                    = _make_collection(ceil,    CeilingHeight(),                       "m")

    epw.save(out_path)
    print(f"  EPW written: {out_path}")


# ---------------------------------------------------------------------------
# 7.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_amy_epw(station_id: str, year: int,
                  api_key: str, email: str,
                  out_path: str):
    """
    Full pipeline: ISD station ID → AMY EPW file.

    Parameters
    ----------
    station_id : str
        NOAA ISD station ID in "USAF-WBAN" format, e.g. "726300-14733"
    year : int
        Target year, e.g. 2023
    api_key : str
        NREL Developer Network API key (free registration at developer.nrel.gov)
    email : str
        Email address for NREL API
    out_path : str
        Output EPW file path
    """
    print(f"\n=== Building AMY EPW: station={station_id}, year={year} ===\n")

    # Step 1: station metadata
    print("[1/5] Station metadata")
    meta = fetch_station_metadata(station_id)

    # Step 2: ISD raw data
    print("\n[2/5] ISD hourly data")
    df_raw = fetch_isd_hourly(meta, year)

    # Step 3: parse + resample to clean hourly
    print("\n[3/5] Parsing + resampling")
    met = process_isd_to_hourly(df_raw, year, meta["tz_offset"])

    # Step 4: gap filling
    print("\n[4/5] Gap filling")
    met = fill_gaps(met)

    # Step 5: NSRDB solar
    print("\n[5/5] Solar data (NSRDB PSM4)")
    solar = fetch_solar_psm4(meta["lat"], meta["lon"], year, api_key, email)

    # Write EPW
    print("\n[6/6] Writing EPW")
    write_epw(met, solar, meta, year, out_path)

    print(f"\n✓ Done → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an AMY EPW from a NOAA ISD station ID"
    )
    parser.add_argument(
        "--station", required=True,
        help='ISD station ID in "USAF-WBAN" format, e.g. 726300-14733'
    )
    parser.add_argument(
        "--year", type=int, default=2023,
        help="Target year (default: 2023)"
    )
    parser.add_argument(
        "--api_key", required=True,
        help="NREL Developer Network API key"
    )
    parser.add_argument(
        "--email", required=True,
        help="Email address for NREL API registration"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output EPW path (default: {station}_{year}.epw)"
    )
    args = parser.parse_args()

    out = args.out or f"{args.station.replace('-','_')}_{args.year}.epw"
    build_amy_epw(args.station, args.year, args.api_key, args.email, out)
