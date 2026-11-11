"""beijing_loader.py - load the UCI Beijing multi-site air-quality data.

Usage (PowerShell, in the folder holding this file):
    python beijing_loader.py "D:\\...\\Dataset\\beijing+multi+site+air+quality+data.zip"

Needs:  python -m pip install pandas
It only READS the zip. Nothing is written unless you pass --save.
"""
import io
import re
import sys
import zipfile

import pandas as pd

MISSING = ["NA", "", "nan", "-200", "-9999"]
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
# Plausible physical ranges. Values outside become missing (NaN).
RANGES = {
    "PM2.5": (0, 1500), "PM10": (0, 1500), "SO2": (0, 1000),
    "NO2": (0, 1000), "CO": (0, 20000), "O3": (0, 1000),
    "TEMP": (-50, 50), "PRES": (900, 1100), "DEWP": (-60, 40),
    "RAIN": (0, 300), "WSPM": (0, 60),
}


def _csv_files(zf):
    """Yield (name, bytes) for every station CSV, even inside a nested zip."""
    for info in zf.infolist():
        low = info.filename.lower()
        if info.is_dir():
            continue
        if low.endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            yield from _csv_files(inner)
        elif low.endswith(".csv") and "prsa_data_" in low:
            yield info.filename, zf.read(info)   # skips data.csv / test.csv


def load_beijing_multisite(zip_path, apply_ranges=True):
    """Return one DataFrame: all stations, hourly, indexed by (station, time).
    df.attrs["removed"] = how many values the range limits turned into NaN."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name, raw in _csv_files(zf):
            df = pd.read_csv(io.BytesIO(raw), na_values=MISSING,
                             keep_default_na=True)
            frames.append(df)
    if not frames:
        raise ValueError("No PRSA_Data_*.csv files found in the zip.")
    df = pd.concat(frames, ignore_index=True)

    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"])

    removed = {}
    for col, (lo, hi) in RANGES.items():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            bad = (df[col] < lo) | (df[col] > hi)
            removed[col] = int(bad.sum())
            if apply_ranges:
                df.loc[bad, col] = float("nan")

    df = df.sort_values(["station", "time"]).set_index(["station", "time"])
    df.attrs["removed"] = removed
    return df


def quality_report(df):
    """Small table: rows and % missing per station and column."""
    cols = [c for c in POLLUTANTS + WEATHER if c in df]
    miss = df[cols].isna().groupby(level="station").mean().mul(100).round(1)
    miss.insert(0, "rows", df.groupby(level="station").size())
    return miss


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = load_beijing_multisite(sys.argv[1])
    print(f"Stations: {data.index.get_level_values('station').nunique()}")
    print(f"Rows: {len(data):,}")
    t = data.index.get_level_values("time")
    print(f"Time range: {t.min()}  to  {t.max()}\n")
    print("Missing % per station:")
    print(quality_report(data).to_string())
    if "--save" in sys.argv:
        data.to_csv("beijing_clean.csv")
        print("\nSaved beijing_clean.csv")
