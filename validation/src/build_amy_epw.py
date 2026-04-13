"""
build_amy_epw.py
----------------
AMYEPWBuilder — builds an Actual Meteorological Year (AMY) EPW file from
a NOAA ISD station ID and NSRDB solar data.

Intended as one component in the NEEA → energy model conversion pipeline:

    builder = AMYEPWBuilder(
        station_id="726300-14733",
        year=2023,
        api_key="YOUR_NREL_KEY",
        email="you@example.com",
    )
    epw_path = builder.build("portland_2023.epw")

    # Intermediate results are available after build():
    builder.station_meta   # dict: lat, lon, elev_m, tz_offset, ...
    builder.met            # pd.DataFrame: cleaned 8760-row met data
    builder.solar          # pd.DataFrame: ghi, dni, dhi (W/m²)

Pipeline steps (also callable individually):
    1. fetch_station_metadata()   — ISD history → lat/lon/elev/tz
    2. fetch_met()                — ISD global-hourly CSV → hourly DataFrame
    3. fetch_solar()              — NSRDB PSM4 → DNI/DHI/GHI
    4. build(out_path)            — runs all steps, writes EPW

Requirements
------------
    pip install requests pandas numpy pvlib ladybug-core
"""

import io
import logging
import numpy as np
import pandas as pd
import pvlib
import requests
# ladybug is only used to read back / verify the written EPW; writing is done directly.

log = logging.getLogger(__name__)


class AMYEPWBuilder:
    """
    Builds an Actual Meteorological Year (AMY) EPW file for a given
    NOAA ISD station and year.

    Parameters
    ----------
    station_id : str
        NOAA ISD station in "USAF-WBAN" format, e.g. "726300-14733".
    year : int
        Target year, e.g. 2023.
    api_key : str
        NREL Developer Network API key (free at developer.nrel.gov).
    email : str
        Email address associated with the NREL API key.
    short_gap_hours : int
        Maximum gap (hours) filled by linear interpolation; longer gaps
        fall back to climatological hourly means. Default: 3.
    """

    _ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    _ISD_DATA_URL    = (
        "https://www.ncei.noaa.gov/data/global-hourly/access"
        "/{year}/{usaf}{wban}.csv"
    )

    def __init__(
        self,
        station_id: str,
        year: int,
        api_key: str,
        email: str,
        short_gap_hours: int = 3,
    ):
        self.station_id      = station_id
        self.year            = year
        self.api_key         = api_key
        self.email           = email
        self.short_gap_hours = short_gap_hours

        # Populated incrementally; available for inspection after each step
        self.station_meta: dict | None        = None
        self.met:          pd.DataFrame | None = None
        self.solar:        pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public pipeline entry point
    # ------------------------------------------------------------------

    def build(self, out_path: str | None = None) -> str:
        """
        Run the full pipeline and write an EPW file.

        Parameters
        ----------
        out_path : str, optional
            Destination path for the EPW file.  Defaults to
            ``{station_id}_{year}.epw`` in the current directory.

        Returns
        -------
        str
            Absolute path of the written EPW file.
        """
        if out_path is None:
            out_path = f"{self.station_id.replace('-', '_')}_{self.year}.epw"

        log.info("=== AMYEPWBuilder: station=%s, year=%d ===", self.station_id, self.year)

        log.info("[1/4] Fetching station metadata")
        self.fetch_station_metadata()

        log.info("[2/4] Fetching + processing ISD met data")
        self.fetch_met()

        log.info("[3/4] Fetching NSRDB solar data")
        self.fetch_solar()

        log.info("[4/4] Writing EPW → %s", out_path)
        self._write_epw(out_path)

        log.info("Done → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Step 1 — ISD station metadata
    # ------------------------------------------------------------------

    def fetch_station_metadata(self) -> dict:
        """
        Download the ISD history file and parse metadata for this station.

        Populates ``self.station_meta`` and returns it.

        Returns
        -------
        dict
            Keys: lat, lon, elev_m, name, state, country, tz_offset, usaf, wban
        """
        usaf, wban = self.station_id.split("-")
        log.debug("Downloading ISD history from %s", self._ISD_HISTORY_URL)

        resp = requests.get(self._ISD_HISTORY_URL, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), dtype=str)
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

        row = df[(df["USAF"] == usaf) & (df["WBAN"] == wban)]
        if row.empty:
            raise ValueError(
                f"Station {self.station_id} not found in ISD history. "
                "Verify the USAF and WBAN portions."
            )
        row = row.iloc[0]

        lat  = float(row["LAT"])
        lon  = float(row["LON"])
        elev = float(row["ELEV(M)"]) if "ELEV(M)" in row.index else 0.0

        self.station_meta = dict(
            lat=lat, lon=lon, elev_m=elev,
            name=row.get("STATION_NAME", self.station_id),
            state=row.get("STATE", ""),
            country=row.get("CTRY", ""),
            tz_offset=round(lon / 15),   # rough UTC offset; refine if needed
            usaf=usaf, wban=wban,
        )
        log.info(
            "Station: %s  (%.4f, %.4f)  elev=%.0f m  UTC%+d",
            self.station_meta["name"], lat, lon, elev,
            self.station_meta["tz_offset"],
        )
        return self.station_meta

    # ------------------------------------------------------------------
    # Step 2 — ISD met data (download → parse → resample → gap-fill)
    # ------------------------------------------------------------------

    def fetch_met(self) -> pd.DataFrame:
        """
        Download ISD global-hourly data, parse, resample to clean 8760-row
        hourly DataFrame, and gap-fill.

        Requires ``self.station_meta`` (call ``fetch_station_metadata()`` first
        or use ``build()`` which handles ordering automatically).

        Populates ``self.met`` and returns it.
        """
        if self.station_meta is None:
            raise RuntimeError("Call fetch_station_metadata() before fetch_met().")

        meta = self.station_meta
        url  = self._ISD_DATA_URL.format(
            year=self.year, usaf=meta["usaf"], wban=meta["wban"].zfill(5)
        )
        log.debug("Downloading ISD data: %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        df_raw = pd.read_csv(
            io.StringIO(resp.text), low_memory=False, parse_dates=["DATE"]
        )
        df_raw = df_raw.rename(columns={"DATE": "utc_time"})
        df_raw = df_raw.set_index("utc_time").sort_index()
        log.info("ISD: %d raw observations downloaded", len(df_raw))

        met = self._parse_isd_to_hourly(df_raw)
        met = self._fill_gaps(met)

        self.met = met
        return self.met

    def _parse_isd_to_hourly(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Parse ISD raw fields and resample to an 8760-row local-time DataFrame."""
        out = pd.DataFrame(index=df_raw.index)

        if "TMP" in df_raw.columns:
            out["temp_C"]  = self._parse_field(df_raw["TMP"], 0.1,   9999)
        if "DEW" in df_raw.columns:
            out["dewp_C"]  = self._parse_field(df_raw["DEW"], 0.1,   9999)
        if "SLP" in df_raw.columns:
            slp_hpa        = self._parse_field(df_raw["SLP"], 0.1,  99999)
            out["pressure_Pa"] = slp_hpa * 100.0
        if "WND" in df_raw.columns:
            out["wind_dir_deg"], out["wind_speed_ms"] = self._parse_wind(df_raw["WND"])
        if "CIG" in df_raw.columns:
            out["ceiling_m"]    = self._parse_field(df_raw["CIG"], 1.0, 99999)
        if "VIS" in df_raw.columns:
            out["visibility_m"] = self._parse_field(df_raw["VIS"], 1.0, 999999)

        if "GD1" in df_raw.columns:
            out["total_sky_cover"] = df_raw["GD1"].apply(self._parse_sky_cover)
        else:
            log.warning("ISD: GD1 cloud field not present — total_sky_cover will default to 5 (50%%) in EPW")
            out["total_sky_cover"] = np.nan
        out["precip_mm"] = (
            self._parse_field(df_raw["AA1"], 0.1, 9999)
            if "AA1" in df_raw.columns
            else 0.0
        )

        # Resample to hourly (last valid obs per UTC hour)
        out = out[out.index.year == self.year]
        out_hourly = out.resample("1h").last()

        # Build 8760-row UTC grid aligned so that after the local-time shift
        # the index runs exactly Jan 1 00:00 → Dec 31 23:00 of the target year.
        # local = utc + tz_offset  →  utc_start = Jan 1 00:00 local − tz_offset
        tz = self.station_meta["tz_offset"]
        utc_start = pd.Timestamp(f"{self.year}-01-01 00:00", tz="UTC") - pd.Timedelta(hours=tz)
        full_utc = pd.date_range(start=utc_start, periods=8760, freq="1h", tz="UTC")

        out_hourly.index = pd.DatetimeIndex(out_hourly.index).tz_localize("UTC")
        out_hourly = out_hourly.reindex(full_utc)

        # Convert to local standard time
        out_hourly.index = (out_hourly.index + pd.Timedelta(hours=tz)).tz_localize(None)

        # Derived relative humidity
        if "temp_C" in out_hourly and "dewp_C" in out_hourly:
            T, Td = out_hourly["temp_C"], out_hourly["dewp_C"]
            out_hourly["rh_pct"] = (
                100 * np.exp((17.625 * Td) / (243.04 + Td))
                    / np.exp((17.625 * T)  / (243.04 + T))
            ).clip(0, 100)

        log.info("ISD: resampled to %d hourly rows (local time)", len(out_hourly))
        return out_hourly

    # ------------------------------------------------------------------
    # Step 3 — NSRDB solar data
    # ------------------------------------------------------------------

    def fetch_solar(self) -> pd.DataFrame:
        """
        Retrieve actual-year hourly GHI/DNI/DHI from NSRDB PSM4.

        Requires ``self.station_meta``.

        Populates ``self.solar`` and returns it.
        """
        if self.station_meta is None:
            raise RuntimeError("Call fetch_station_metadata() before fetch_solar().")

        lat, lon = self.station_meta["lat"], self.station_meta["lon"]
        log.info("NSRDB PSM4: lat=%.4f, lon=%.4f, year=%d", lat, lon, self.year)

        solar_df, meta = pvlib.iotools.get_nsrdb_psm4_aggregated(
            latitude=lat,
            longitude=lon,
            api_key=self.api_key,
            email=self.email,
            year=self.year,
            time_step=60,
            parameters=("ghi", "dni", "dhi", "surface_pressure",
                         "air_temperature", "wind_speed"),
            map_variables=True,
        )
        log.info("NSRDB: %d rows, TZ offset=%s", len(solar_df), meta.get("Time Zone", "?"))

        self.solar = solar_df[["ghi", "dni", "dhi"]].copy()
        self.solar.index = self.solar.index.tz_localize(None)
        return self.solar

    # ------------------------------------------------------------------
    # Step 4 — Write EPW
    # ------------------------------------------------------------------

    def _write_epw(self, out_path: str) -> None:
        """
        Merge met + solar and write an EPW file directly in the EnergyPlus text format.

        EPW is a comma-separated format: 8 header lines followed by 8760 data rows,
        each with 35 fields. Writing directly avoids relying on ladybug internals
        that change across versions.
        """
        if self.met is None or self.solar is None or self.station_meta is None:
            raise RuntimeError("Run fetch_met() and fetch_solar() before writing EPW.")

        m = self.station_meta

        # Align both DataFrames to the same 8760-row local-time index
        local_index = pd.date_range(
            start=f"{self.year}-01-01 00:00", periods=8760, freq="1h"
        )

        def _align(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.index = pd.DatetimeIndex(df.index)
            if len(df) != 8760:
                df = df.reindex(local_index, method="nearest", tolerance="1h")
            else:
                df.index = local_index
            return df

        met   = _align(self.met)
        solar = _align(self.solar)

        def _get(df, col, default):
            return df[col].fillna(default).values if col in df.columns else np.full(8760, default)

        temp_C = _get(met, "temp_C",          20.0)
        dewp_C = _get(met, "dewp_C",          10.0)
        rh     = _get(met, "rh_pct",          50.0)
        press  = _get(met, "pressure_Pa",     101325.0)
        wd     = _get(met, "wind_dir_deg",    0.0)
        ws     = _get(met, "wind_speed_ms",   0.0)
        sky    = _get(met, "total_sky_cover",  5.0)
        ceil_  = _get(met, "ceiling_m",       99999.0)
        vis    = _get(met, "visibility_m",    9999.0)
        ghi    = _get(solar, "ghi", 0.0).clip(0)
        dni    = _get(solar, "dni", 0.0).clip(0)
        dhi    = _get(solar, "dhi", 0.0).clip(0)
        ir     = self._calc_horizontal_ir(
            pd.Series(temp_C), pd.Series(dewp_C), pd.Series(sky)
        ).values

        # Extraterrestrial radiation — above-atmosphere values, purely geometric.
        # EnergyPlus uses these as the theoretical solar upper bound (clearness index, daylighting).
        # Computed analytically from pvlib: no measurement data needed.
        tz_str  = f"Etc/GMT{m['tz_offset']:+d}".replace("+-", "-")
        times   = pd.DatetimeIndex(local_index).tz_localize(tz_str)
        dni_et  = pvlib.irradiance.get_extra_radiation(times).values           # W/m²
        sol_pos = pvlib.solarposition.get_solarposition(times, m["lat"], m["lon"])
        cos_sza = np.cos(np.radians(sol_pos["apparent_zenith"].values)).clip(0)
        ghi_et  = (dni_et * cos_sza).clip(0)                                   # W/m²

        # Illuminance derived from irradiance using standard luminous efficacy.
        # ~110 lm/W is the EnergyPlus default for solar radiation.
        # Zenith luminance requires a sky radiance model and is left missing.
        LUM_EFFICACY = 110.0  # lm/W
        ghi_illum = (ghi * LUM_EFFICACY).astype(int)   # lux
        dni_illum = (dni * LUM_EFFICACY).astype(int)   # lux
        dhi_illum = (dhi * LUM_EFFICACY).astype(int)   # lux

        # ── Header ────────────────────────────────────────────────────────────
        header = "\n".join([
            f"LOCATION,{m['name']},{m['state']},{m['country']},"
            f"NOAA-ISD+NSRDB-PSM4 AMY,{self.station_id},"
            f"{m['lat']:.2f},{m['lon']:.2f},{m['tz_offset']},{m['elev_m']:.0f}",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            f"COMMENTS 1,AMY EPW | station {self.station_id} | year {self.year} | "
            "NOAA ISD + NSRDB PSM4",
            "COMMENTS 2,Generated by AMYEPWBuilder",
            "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31",
        ]) + "\n"

        # ── Data rows ─────────────────────────────────────────────────────────
        # EPW hours are 1-24 (hour 1 = period ending at 1 AM on that date)
        rows = []
        for i in range(8760):
            dt = local_index[i]
            epw_hour = dt.hour + 1   # 0→1, …, 23→24
            rows.append(",".join([
                str(self.year),           # 0  Year
                str(dt.month),            # 1  Month
                str(dt.day),              # 2  Day
                str(epw_hour),            # 3  Hour (1-24)
                "0",                      # 4  Minute
                "?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0",  # 5 Uncertainty flags
                f"{temp_C[i]:.1f}",       # 6  Dry Bulb Temp (°C)
                f"{dewp_C[i]:.1f}",       # 7  Dew Point Temp (°C)
                str(int(round(rh[i]))),   # 8  Relative Humidity (%)
                str(int(round(press[i]))),# 9  Atmospheric Pressure (Pa)
                str(int(round(ghi_et[i]))),  # 10 Extraterrestrial Horizontal Radiation (Wh/m²)
                str(int(round(dni_et[i]))),  # 11 Extraterrestrial Direct Normal Radiation (Wh/m²)
                str(int(round(ir[i]))),   # 12 Horizontal Infrared Radiation (W/m²)
                str(int(round(ghi[i]))),  # 13 GHI (Wh/m²)
                str(int(round(dni[i]))),  # 14 DNI (Wh/m²)
                str(int(round(dhi[i]))),  # 15 DHI (Wh/m²)
                str(ghi_illum[i]),        # 16 GHI Illuminance (lux)
                str(dni_illum[i]),        # 17 DNI Illuminance (lux)
                str(dhi_illum[i]),        # 18 DHI Illuminance (lux)
                "9999",                   # 19 Zenith Luminance (missing — requires sky model)
                str(int(round(wd[i]))),   # 20 Wind Direction (deg)
                f"{ws[i]:.1f}",           # 21 Wind Speed (m/s)
                str(int(round(min(max(sky[i], 0), 10)))),  # 22 Total Sky Cover (tenths)
                str(int(round(min(max(sky[i], 0), 10)))),  # 23 Opaque Sky Cover (tenths)
                f"{vis[i] / 1000:.1f}",   # 24 Visibility (km)
                str(int(ceil_[i])),       # 25 Ceiling Height (m)
                "9",                      # 26 Present Weather Observation (missing)
                "999999999",              # 27 Present Weather Codes (missing)
                "999",                    # 28 Precipitable Water (missing)
                "999",                    # 29 Aerosol Optical Depth (missing)
                "999",                    # 30 Snow Depth (missing)
                "99",                     # 31 Days Since Last Snowfall (missing)
                "999",                    # 32 Albedo (missing)
                "0",                      # 33 Liquid Precipitation Depth (mm)
                "99",                     # 34 Liquid Precipitation Quantity (missing)
            ]))

        with open(out_path, "w") as f:
            f.write(header)
            f.write("\n".join(rows) + "\n")

        log.info("EPW written: %s", out_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-pass gap filling per column:
          Pass 1 — linear interpolation for gaps <= short_gap_hours
          Pass 2 — climatological mean (hour-of-day × month) for longer gaps
        """
        df = df.copy()
        for col in df.columns:
            n_before = df[col].isna().sum()
            if n_before == 0:
                continue

            df[col] = df[col].interpolate(
                method="linear", limit=self.short_gap_hours, limit_direction="both"
            )
            still_nan = df[col].isna()
            if still_nan.any():
                clim = df[col].groupby([df.index.month, df.index.hour]).transform("mean")
                df.loc[still_nan, col] = clim[still_nan]

            n_after = df[col].isna().sum()
            log.debug(
                "Gap-fill %s: filled %d/%d (remain %d)",
                col, n_before - n_after, n_before, n_after,
            )
        return df

    @staticmethod
    def _parse_field(series: pd.Series, scale: float, sentinel) -> pd.Series:
        """Extract the first comma-delimited numeric value, apply scale factor."""
        def _parse(val):
            if pd.isna(val):
                return np.nan
            try:
                v = float(str(val).split(",")[0])
                return np.nan if v == sentinel else v * scale
            except ValueError:
                return np.nan
        return series.apply(_parse)

    @staticmethod
    def _parse_wind(series: pd.Series):
        """Parse ISD WND field → (wind_dir_deg, wind_speed_ms)."""
        def _parse(val):
            if pd.isna(val):
                return np.nan, np.nan
            parts = str(val).split(",")
            try:
                wd = float(parts[0]); ws = float(parts[3]) * 0.1
                return (np.nan if wd == 999 else wd), (np.nan if ws * 10 == 9999 else ws)
            except (IndexError, ValueError):
                return np.nan, np.nan
        parsed = series.apply(_parse)
        return parsed.apply(lambda x: x[0]), parsed.apply(lambda x: x[1])

    @staticmethod
    def _parse_sky_cover(val) -> float:
        """Parse ISD GD1 cloud field → sky cover in tenths (0-10)."""
        if pd.isna(val):
            return np.nan
        try:
            oktas = float(str(val).split(",")[1])
            return np.nan if oktas >= 9 else round(oktas * 10 / 8)
        except (IndexError, ValueError):
            return np.nan

    @staticmethod
    def _calc_horizontal_ir(
        temp_C: pd.Series,
        dewp_C: pd.Series,
        sky_cover: pd.Series,
    ) -> pd.Series:
        """
        Estimate horizontal infrared radiation (W/m²) via Berdahl & Martin (1984):
            IR = ε·σ·T_air⁴,  ε = 0.711 + 0.56·(Td/273.15) + 0.73·(N/10)²
        """
        sigma = 5.6697e-8
        T_K   = temp_C + 273.15
        Td_K  = dewp_C + 273.15
        N     = (sky_cover.fillna(5) / 10.0).clip(0, 1)
        emiss = (0.711 + 0.56 * (Td_K / 273.15 - 1) + 0.73 * N**2).clip(0, 1)
        return (emiss * sigma * T_K**4).clip(0)
