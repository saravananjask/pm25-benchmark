"""loaders.py - one loader per dataset. Each returns a clean pandas DataFrame
and stores how many values were removed in df.attrs["removed"].

Needs:  python -m pip install pandas pyarrow
Nothing here writes to your data folder.
"""
import glob
import gzip
import os
import zipfile

import numpy as np
import pandas as pd

MISSING = ["NA", "", "nan", "-200", "-9999"]

# Physical limits. A value outside becomes NaN (and is counted).
RANGES = {
    "PM2.5": (0, 1500), "PM10": (0, 1500), "SO2": (0, 1000), "NO2": (0, 1000),
    "CO": (0, 20000), "O3": (0, 1000), "TEMP": (-50, 50), "PRES": (900, 1100),
    "DEWP": (-60, 40), "HUMI": (0, 100), "RAIN": (0, 300),
    "precipitation": (0, 500),            # removes the 999990 error code
}


def _clean(df, ranges=RANGES):
    removed = {}
    for col, (lo, hi) in ranges.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                removed[col] = int(bad.sum())
                df.loc[bad, col] = np.nan
    df.attrs["removed"] = removed
    return df


def _member(zf, ends_with):
    for n in zf.namelist():
        if n.lower().endswith(ends_with.lower()) and "__macosx" not in n.lower():
            return n
    raise FileNotFoundError(f"no file ending '{ends_with}' in the zip")


def _hourly_time(df):
    t = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df.drop(columns=["No", "year", "month", "day", "hour"],
                   errors="ignore").assign(time=t)


# 1. Beijing PM2.5 (UCI id 381), hourly, 2010-2014 ---------------------------
def load_beijing_pm25(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(zf.open(_member(zf, ".csv")), na_values=MISSING)
    df = _hourly_time(df).rename(columns={"pm2.5": "PM2.5"})
    removed_before = None
    df = _clean(df).set_index("time").sort_index()
    return df


# 2. Five Chinese Cities (UCI id 394), hourly, 2010-2015 ---------------------
def load_five_cities(folder, max_missing_pct=30):
    """`folder` = the extracted FiveCitiePMData folder (or its parent).
    The US Post monitors only started reporting in 2012-2013 in four of the five
    cities, so any city-year whose PM2.5 is more than `max_missing_pct` % missing
    is dropped. The dropped city-years are listed in df.attrs["dropped_city_years"]."""
    files = sorted(glob.glob(os.path.join(folder, "**", "*PM20100101_20151231.csv"),
                             recursive=True))
    if not files:
        raise FileNotFoundError("no *PM20100101_20151231.csv files under " + folder)
    frames, removed = [], {}
    for f in files:
        city = os.path.basename(f).split("PM2010")[0]
        df = pd.read_csv(f, na_values=MISSING)
        df = _hourly_time(df)
        others = [c for c in df if c.startswith("PM_") and c != "PM_US Post"]
        df["PM2.5"] = df["PM_US Post"] if "PM_US Post" in df else np.nan
        df["PM2.5_other_monitors"] = df[others].mean(axis=1) if others else np.nan
        df = df.drop(columns=[c for c in df if c.startswith("PM_")])
        df = _clean(df)
        for k, v in df.attrs["removed"].items():
            removed[k] = removed.get(k, 0) + v
        df.insert(0, "city", city)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).set_index(["city", "time"]).sort_index()
    city = out.index.get_level_values("city")
    year = out.index.get_level_values("time").year
    pct = out["PM2.5"].isna().groupby([city, year]).mean() * 100
    bad = pct[pct > max_missing_pct]
    drop = pd.MultiIndex.from_arrays([city, year]).isin(bad.index)
    out = out[~drop]
    out.attrs["removed"] = removed
    out.attrs["dropped_city_years"] = [(c, int(y), round(float(p))) for (c, y), p in bad.items()]
    return out


# 3. Italy Air Quality (UCI id 360), hourly, 2004-2005 -----------------------
def load_italy(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(zf.open(_member(zf, "AirQualityUCI.csv")),
                         sep=";", decimal=",")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(subset=["Date", "Time"])                 # the blank rows
    df["time"] = pd.to_datetime(df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
                                format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).set_index("time").sort_index()
    mask = df == -200
    removed = {c: int(mask[c].sum()) for c in df if mask[c].any()}
    df = df.mask(mask)
    df.attrs["removed"] = removed
    return df


# 4. US EPA AQS daily PM2.5 (parameter 88101) --------------------------------
def load_epa_daily(zip_path, min_obs_percent=75):
    """One value per monitor-day, then the mean over the monitors at a site.
    The file holds several rows for the same monitor-day:
      '24-HR BLK AVG' (standard 24 h average of hourly data), '1 HOUR' (the same
      hourly data aggregated differently) and '24 HOUR' (filter samples).
    We keep ONE row per monitor-day: 24-HR BLK AVG, else 24 HOUR, else 1 HOUR,
    and drop rows with under `min_obs_percent` % hourly completeness.
    `event` marks site-days influenced by an exceptional event (e.g. wildfire smoke)."""
    use = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude",
           "Date Local", "Sample Duration", "Event Type", "Observation Percent",
           "Arithmetic Mean", "State Name"]
    with zipfile.ZipFile(zip_path) as zf:
        df = pd.read_csv(zf.open(_member(zf, ".csv")), usecols=use,
                         dtype={"State Code": str, "County Code": str, "Site Num": str})
    removed = {"rows read": len(df)}
    df["site"] = (df["State Code"].str.zfill(2) + "-" + df["County Code"].str.zfill(3)
                  + "-" + df["Site Num"].str.zfill(4))
    df["date"] = pd.to_datetime(df["Date Local"])
    low = df["Observation Percent"] < min_obs_percent
    removed[f"rows under {min_obs_percent}% completeness"] = int(low.sum())
    df = df[~low]
    prio = {"24-HR BLK AVG": 1, "24 HOUR": 2, "1 HOUR": 3}
    df = df.assign(prio=df["Sample Duration"].map(prio).fillna(9))
    n0 = len(df)
    df = df.sort_values("prio", kind="stable").drop_duplicates(["site", "date", "POC"])
    removed["duplicate-duration rows dropped"] = n0 - len(df)
    neg = df["Arithmetic Mean"] < 0
    removed["negative monitor-days set to NaN"] = int(neg.sum())
    df.loc[neg, "Arithmetic Mean"] = np.nan
    df["event"] = df["Event Type"].isin(["Included", "Excluded"])
    out = df.groupby(["site", "date"]).agg(**{
        "PM2.5": ("Arithmetic Mean", "mean"), "n_monitors": ("Arithmetic Mean", "count"),
        "event": ("event", "max"), "lat": ("Latitude", "first"),
        "lon": ("Longitude", "first"), "state": ("State Name", "first")})
    out.attrs["removed"] = removed
    return out


# 5. India CPCB (Kaggle archive.zip) -----------------------------------------
def load_india(zip_path, table="city_day"):
    """table: city_day, station_day, city_hour or station_hour (hour = large)."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist()
                    if os.path.basename(n).lower() == f"{table}.csv")
        df = pd.read_csv(zf.open(name), na_values=MISSING)
    key = "City" if "City" in df else "StationId"
    tcol = "Datetime" if "Datetime" in df else "Date"
    df["time"] = pd.to_datetime(df[tcol])
    df = _clean(df).drop(columns=[tcol])
    return df.set_index([key, "time"]).sort_index()


def load_india_stations(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_csv(zf.open(_member(zf, "stations.csv")), encoding="utf-8-sig")


# 6. OpenAQ archive (folder of .csv.gz) --------------------------------------
OPENAQ_NAMES = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "co": "CO",
                "so2": "SO2", "o3": "O3"}
MOLAR_MASS = {"co": 28.01, "no2": 46.0055, "o3": 47.9982, "so2": 64.066}  # g/mol
MOLAR_VOLUME = 24.45          # litres/mol at 25 degC and 1 atm


def load_openaq(folder, convert_units=True, full_grid=True):
    """Hourly wide table per location, in UTC.
    convert_units: gases reported in ppm/ppb are converted to ug/m3 (25 degC, 1 atm)
                   so every column matches the other datasets.
    full_grid:     insert the missing hours as NaN rows (so gaps are visible)."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv.gz")))
    if not files:
        raise FileNotFoundError("no .csv.gz files in " + folder)
    df = pd.concat((pd.read_csv(f, usecols=["location_id", "datetime", "parameter",
                                            "units", "value"]) for f in files),
                   ignore_index=True)
    units = df.groupby("parameter")["units"].agg(lambda s: sorted(s.unique())).to_dict()
    neg = df["value"] < 0                                    # -999 etc. = missing
    removed = {"negative values": int(neg.sum())}
    df = df[~neg & df["parameter"].isin(OPENAQ_NAMES)].copy()
    if convert_units:
        u = df["units"].str.lower()
        mw = df["parameter"].map(MOLAR_MASS)
        ppm, ppb = (u == "ppm") & mw.notna(), (u == "ppb") & mw.notna()
        df.loc[ppm, "value"] *= mw[ppm] * 1000 / MOLAR_VOLUME
        df.loc[ppb, "value"] *= mw[ppb] / MOLAR_VOLUME
        removed["values converted ppm/ppb -> ug/m3"] = int(ppm.sum() + ppb.sum())
    df["time"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("h")
    df["parameter"] = df["parameter"].map(OPENAQ_NAMES)
    wide = df.pivot_table(index=["location_id", "time"], columns="parameter",
                          values="value", aggfunc="mean").sort_index()
    if full_grid:
        parts = {}
        for loc, g in wide.groupby(level="location_id"):
            g = g.droplevel("location_id")
            parts[loc] = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        wide = pd.concat(parts, names=["location_id", "time"])
    wide.attrs["removed"] = removed
    wide.attrs["units"] = units
    return wide


# 7. EEA annual station averages (Parquet) -----------------------------------
def load_eea_annual(path, min_coverage=75):
    """The EEA file only publishes station-years with >= 75% data coverage
    (cov.year.* is a fraction, min 0.75), so the filter below is a safety net."""
    df = pd.read_parquet(path).rename(columns={"Air.Quality.Station.EoI.Code": "station",
                                               "Countrycode": "country"})
    df = df.drop(columns=["geometry"], errors="ignore")
    df["year"] = df["year"].astype(int)
    removed = {}
    for p in ["NO2", "O3", "PM10", "PM2.5", "SO2"]:
        if p not in df:
            continue
        neg = df[p] < 0
        low = pd.Series(False, index=df.index)
        cov = f"cov.year.{p}"
        if cov in df:
            c = df[cov] * (100 if df[cov].max() <= 1.0 else 1)
            low = c < min_coverage
        removed[p] = int(((neg | low) & df[p].notna()).sum())
        df.loc[neg | low, p] = np.nan
    df.attrs["removed"] = removed
    return df.set_index(["station", "year"]).sort_index()


# 8. NOAA GHCN-Daily (.dly) --------------------------------------------------
def load_ghcn_dly(path, elements=("TAVG", "TMAX", "TMIN", "PRCP")):
    """Daily weather. Values are tenths of degC / mm in the file; returned in degC / mm.
    Days with a quality flag are dropped."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line) < 21 or line[17:21] not in elements:
                continue
            el, year, month = line[17:21], int(line[11:15]), int(line[15:17])
            for d in range(31):
                base = 21 + 8 * d
                value = int(line[base:base + 5])
                if value == -9999 or line[base + 6:base + 7].strip():
                    continue
                try:
                    rows.append((pd.Timestamp(year, month, d + 1), el, value / 10))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=["date", "element", "value"])
    return df.pivot_table(index="date", columns="element", values="value").sort_index()


def add_previous_day(df, daily, prefix="noaa_"):
    """Attach YESTERDAY's daily weather to hourly rows. Same-day daily values
    contain the whole day (including hours after the forecast time), which
    would be future information, so we shift by one day."""
    d = daily.copy()
    d.index = d.index + pd.Timedelta(days=1)
    d.columns = [prefix + c for c in d.columns]
    times = df.index.get_level_values("time") if "time" in df.index.names else df.index
    add = d.reindex(times.floor("D"))
    add.index = df.index
    return pd.concat([df, add], axis=1)


def ghcn_coverage(daily, start, end):
    """% of days in [start, end] that have a valid value, per element."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    w = daily.loc[start:end]
    return (w.notna().sum() / days * 100).round(1)
