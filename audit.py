"""audit.py - load every dataset with its loader and report what is inside.

Usage (PowerShell, in the folder with loaders.py and beijing_loader.py):
    python audit.py "D:\\...\\Dataset"
Writes audit_report.txt next to this script. Read-only on your data.
Needs:  python -m pip install pandas pyarrow
"""
import glob
import os
import sys
import traceback

import pandas as pd

import loaders as L
from beijing_loader import load_beijing_multisite

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def first(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return hits[0] if hits else None


def time_level(df):
    return df.index.get_level_values("time") if "time" in df.index.names else (
        df.index.get_level_values("date") if "date" in df.index.names else None)


def gaps(df, freq):
    """Missing time steps and duplicate timestamps, per group (e.g. per station)."""
    if "time" not in df.index.names and "date" not in df.index.names:
        return None
    lvl = "time" if "time" in df.index.names else "date"
    keys = [n for n in df.index.names if n != lvl]
    miss = dup = 0
    groups = df.groupby(level=keys) if keys else [(None, df)]
    for _, g in groups:
        t = g.index.get_level_values(lvl)
        dup += int(t.duplicated().sum())
        full = pd.date_range(t.min(), t.max(), freq=freq)
        miss += len(full) - t.nunique()
    return miss, dup


def summarize(name, df, freq=None, focus=("PM2.5", "PM10", "CO")):
    out(f"\n=== {name} ===")
    out(f"rows: {len(df):,}   columns: {len(df.columns)}   index: {list(df.index.names)}")
    t = time_level(df)
    if t is not None:
        out(f"time range: {t.min()}  to  {t.max()}")
    if freq:
        r = gaps(df, freq)
        if r:
            out(f"missing time steps (freq {freq}): {r[0]:,}   duplicate timestamps: {r[1]:,}")
    num = df.select_dtypes("number")
    miss = (num.isna().mean() * 100).round(1)
    out("missing %: " + ", ".join(f"{c} {v}" for c, v in miss.items() if c in focus or v > 20))
    first_lvl = df.index.names[0]
    if first_lvl in ("city", "station") and df.index.get_level_values(0).nunique() <= 15:
        col = next((c for c in ("PM2.5",) if c in df), None)
        if col:
            per = (df[col].isna().groupby(level=0).mean() * 100).round(1)
            out(f"{col} missing % per {first_lvl}: " + ", ".join(f"{k} {v}" for k, v in per.items()))
    for key in ("dropped_city_years",):
        if df.attrs.get(key):
            out(f"{key} (city, year, % missing): {df.attrs[key]}")
    removed = df.attrs.get("removed")
    if removed:
        out("values outside range limits (set to NaN when cleaning): " + ", ".join(f"{k} {v:,}" for k, v in removed.items()))
    for c in focus:
        if c in num and num[c].notna().any():
            mx = num[c].max()
            n = int((num[c] == mx).sum())
            out(f"  {c}: max {mx:g} occurs {n:,} times" + ("   <- looks like a cap" if n >= 20 else ""))


def attempt(name, fn, **kw):
    try:
        df = fn()
        summarize(name, df, **kw)
        return df
    except Exception as e:
        out(f"\n=== {name} ===\n!! FAILED: {type(e).__name__}: {e}")
        LINES.append(traceback.format_exc())
        return None


out(f"Auditing: {os.path.abspath(ROOT)}")
p = first("beijing+multi*.zip")
if p:
    raw = attempt("Beijing multi-site (raw, no range limits)",
                  lambda: load_beijing_multisite(p, apply_ranges=False), freq="h")
    attempt("Beijing multi-site (cleaned)", lambda: load_beijing_multisite(p), freq="h")
p = first("beijing+pm2*.zip")
if p:
    attempt("Beijing PM2.5 (2010-2014)", lambda: L.load_beijing_pm25(p), freq="h",
            focus=("PM2.5",))
p = first("pm2+5+data+of+five+chinese+cities")
if p:
    attempt("Five Chinese Cities", lambda: L.load_five_cities(p), freq="h",
            focus=("PM2.5", "precipitation"))
p = first("air+quality.zip")
if p:
    attempt("Italy", lambda: L.load_italy(p), freq="h", focus=("CO(GT)", "C6H6(GT)"))
p = first("daily_88101_*.zip")
if p:
    attempt("EPA daily PM2.5", lambda: L.load_epa_daily(p), freq="D", focus=("PM2.5",))
p = first("archive.zip")
if p:
    attempt("India city_day", lambda: L.load_india(p, "city_day"), freq="D")
    attempt("India station_day", lambda: L.load_india(p, "station_day"), freq="D")
p = first("data")
if p and os.path.isdir(p):
    d = attempt("OpenAQ", lambda: L.load_openaq(p), freq="h")
    if d is not None:
        out("units seen: " + str(d.attrs.get("units")))
p = first("*.parquet")
if p:
    attempt("EEA annual", lambda: L.load_eea_annual(p), focus=("PM2.5", "PM10"))
p = first("*.dly")
if p:
    noaa = attempt("NOAA GHCN (Beijing)", lambda: L.load_ghcn_dly(p), freq="D",
                   focus=("TAVG", "TMAX", "TMIN", "PRCP"))
    if noaa is not None:
        cov = L.ghcn_coverage(noaa, "2013-03-01", "2017-02-28")
        out("NOAA daily coverage in the Beijing multi-site period "
            "(2013-03-01 to 2017-02-28), % of days: " + str(cov.to_dict()))

with open("audit_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES))
out("\nSaved audit_report.txt")
