"""multibench.py - one forecasting protocol, run on every dataset.

Usage (PowerShell, in the folder with loaders.py and beijing_loader.py):
    python multibench.py "D:\\...\\Dataset"
    python multibench.py "D:\\...\\Dataset" --no-forest              (fast)
    python multibench.py "D:\\...\\Dataset" --datasets epa_daily,openaq
    python multibench.py "D:\\...\\Dataset" --rolling 4              (rolling-origin)
    python multibench.py "D:\\...\\Dataset" --tune --rolling 4      (tuned models too)
    python multibench.py --list

Needs:  python -m pip install pandas scikit-learn pyarrow
Writes multibench_results.csv (one row per dataset x group x horizon x model),
multibench_ci.csv (95% block-bootstrap intervals for every gain), with --rolling K
multibench_rolling.csv (K expanding-window folds) and with --tune
multibench_tuned_params.csv (the settings chosen each time).
Reads your data only; nothing in the data folder is changed.

Protocol (the same for every dataset):
  * target = PM2.5 at time t + horizon; features use information up to time t only
  * short gaps are filled by carrying the last value forward (never backward);
    the target is never filled
  * fixed-date split per dataset (cutoff below), never shuffled
  * PURGE GAP: a training row is used only if its TARGET time is before the cutoff
  * missing feature values are imputed with the training median (inside the pipeline,
    fitted on training data only)
  * confidence intervals: moving-block bootstrap over TIME blocks (7 days hourly,
    14 days daily) resampled jointly across all stations/sites, 1000 replicates
  * tuning (--tune): ridge alpha and forest settings are chosen by an INNER time-ordered
    cross-validation that uses the training period only (3 expanding windows, same purge
    gap), then refitted on the whole training period and scored once on the test period.
    Default-setting models ("ridge", "forest") are always reported next to the tuned
    ones ("ridge_tuned", "forest_tuned"). With --rolling, tuning is repeated inside
    every fold, so no fold ever sees its own test window while tuning.
  * rolling origin (--rolling K): K consecutive test windows; each fold trains only on
    data whose targets precede its window (expanding window, same purge gap)
  * baselines: persistence (value now), seasonal-naive (same time of the last cycle;
    equals persistence when the horizon is a whole cycle) and trailing-mean (mean of the
    last 24 hours, or last 7 days for daily data)
"""
import argparse
import glob
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import loaders as L
from beijing_loader import load_beijing_multisite

warnings.filterwarnings("ignore", message="Skipping features without any observed values")

FREQ = {
    "h": dict(step=pd.Timedelta(hours=1), season=24, gap=3, unit="h",
              lags=[0, 1, 2, 3, 6, 12, 24, 48], horizons=[1, 6, 24, 48]),
    "D": dict(step=pd.Timedelta(days=1), season=7, gap=2, unit="d",
              lags=[0, 1, 2, 3, 7, 14], horizons=[1, 3, 7]),
}
MIN_TRAIN, MIN_TEST = 200, 20


def _find(pattern):
    def go(root):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        return hits[0] if hits else None
    return go


def registry():
    return {
        "beijing_multisite": dict(
            find=_find("beijing+multi*.zip"), load=load_beijing_multisite,
            group="station", freq="h", cutoff="2016-05-01",
            features=["PM2.5", "PM10", "NO2", "CO", "TEMP", "PRES", "DEWP", "WSPM"]),
        "beijing_pm25": dict(
            find=_find("beijing+pm2*.zip"), load=L.load_beijing_pm25,
            group=None, freq="h", cutoff="2014-01-01",
            features=["PM2.5", "DEWP", "TEMP", "PRES", "Iws"]),
        "five_cities": dict(
            find=_find("pm2+5+data+of+five+chinese+cities"), load=L.load_five_cities,
            group="city", freq="h", cutoff="2015-01-01",
            features=["PM2.5", "DEWP", "HUMI", "PRES", "TEMP", "Iws", "precipitation"]),
        "india_city_day": dict(
            find=_find("archive.zip"), load=lambda p: L.load_india(p, "city_day"),
            group="City", freq="D", cutoff="2019-06-01", pooled=True,  # per-city models
            # overfit (cities have only 350-1600 training days); test includes 2020 lockdown
            features=["PM2.5", "PM10", "NO2", "CO", "O3"]),
        "epa_daily": dict(
            find=_find("daily_88101_*.zip"), load=L.load_epa_daily,
            group="site", freq="D", cutoff="2023-10-01", pooled=True,
            min_coverage=0.8, features=["PM2.5"], static=["lat", "lon"]),
        "openaq": dict(
            find=_find("data"), load=L.load_openaq,
            group="location_id", freq="h", cutoff="2020-10-01",
            features=["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]),
    }


def iter_groups(df, group):
    """Yield (name, frame indexed by naive-UTC time)."""
    parts = [("all", df)] if group is None else [
        (k, g.droplevel(group)) for k, g in df.groupby(level=group)]
    for k, g in parts:
        if getattr(g.index, "tz", None) is not None:
            g = g.copy()
            g.index = g.index.tz_convert("UTC").tz_localize(None)
        yield k, g[~g.index.duplicated()].sort_index()


def time_features(idx, freq):
    if freq == "h":
        a, b = 2 * np.pi * idx.hour / 24, 2 * np.pi * idx.month / 12
    else:
        a, b = 2 * np.pi * idx.dayofweek / 7, 2 * np.pi * idx.dayofyear / 366
    return pd.DataFrame({"t1_sin": np.sin(a), "t1_cos": np.cos(a),
                         "t2_sin": np.sin(b), "t2_cos": np.cos(b)}, index=idx)


def build(g, cfg, horizon):
    """(X, y, seasonal_baseline) for one group. Row t only knows the past up to t."""
    f = FREQ[cfg["freq"]]
    target, feats = "PM2.5", [c for c in cfg["features"] if c in g]
    static = {s: g[s].dropna().iloc[0] if g[s].notna().any() else np.nan
              for s in cfg.get("static", [])}
    raw = g.asfreq(cfg["freq"])
    d = raw.ffill(limit=f["gap"])                       # causal gap filling
    X = pd.concat({f"{c}_lag{lag}": d[c].shift(lag) for c in feats for lag in f["lags"]},
                  axis=1)
    X = pd.concat([X, time_features(d.index, cfg["freq"])], axis=1)
    for s, v in static.items():
        X[s] = v
    y = raw[target].shift(-horizon)                      # raw target, never filled
    k = math.ceil(horizon / f["season"])
    base = d[target].shift(f["season"] * k - horizon)    # same time, last cycle
    roll = d[target].rolling(f["season"], min_periods=max(1, f["season"] // 2)).mean()
    B = pd.DataFrame({"seasonal_naive": base, "trailing_mean": roll})
    keep = y.notna() & X[f"{target}_lag0"].notna() & B.notna().all(axis=1)
    return X[keep], y[keep], B[keep]


def split(X, y, base, cutoff, horizon, step, end=None):
    cut = pd.Timestamp(cutoff)
    tr = X.index <= cut - (horizon + 1) * step           # purge gap
    te = X.index >= cut
    if end is not None:
        te = te & (X.index < pd.Timestamp(end))
    return X[tr], X[te], y[tr], y[te], base[te]


def ridge_model(alpha=1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=alpha))


def forest_model(min_samples_leaf=5, max_features=1.0, n_estimators=100):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=n_estimators,
                                               min_samples_leaf=min_samples_leaf,
                                               max_features=max_features,
                                               n_jobs=-1, random_state=0))


def make_models(forest):
    models = {"ridge": ridge_model()}
    if forest:
        models["forest"] = forest_model()
    return models


RIDGE_GRID = [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)]
FOREST_GRID = [{"min_samples_leaf": leaf, "max_features": mf}
               for leaf in (5, 20) for mf in (1.0, 0.33)]
INNER_FOLDS = 3
TUNE_TREES = 40                  # fewer trees while searching, 100 for the final fit


def inner_folds(times, horizon, step, n=INNER_FOLDS):
    """Expanding-window folds INSIDE the training period (time-ordered, purged).
    Yields (train_positions, validation_positions)."""
    u = pd.DatetimeIndex(times.unique()).sort_values()
    bounds = [u[int(len(u) * k / (n + 1))] for k in range(1, n + 1)] + [u[-1] + step]
    for i in range(n):
        v0, v1 = bounds[i], bounds[i + 1]
        tr = np.where(times <= v0 - (horizon + 1) * step)[0]
        va = np.where((times >= v0) & (times < v1))[0]
        yield tr, va


def tune(factory, grid, Xtr, ytr, horizon, step):
    """Pick the setting with the lowest mean validation MAE over the inner folds.
    Falls back to the first (default-like) setting when the training period is too
    short for inner folds."""
    times, best = Xtr.index, (None, np.inf)
    for params in grid:
        errs = []
        for tr, va in inner_folds(times, horizon, step):
            if len(tr) < 100 or len(va) < 20:
                continue
            m = factory(**params).fit(Xtr.iloc[tr], ytr.iloc[tr])
            errs.append(mean_absolute_error(ytr.iloc[va], m.predict(Xtr.iloc[va])))
        if errs and np.mean(errs) < best[1]:
            best = (params, float(np.mean(errs)))
    return best if best[0] is not None else (grid[0], float("nan"))


def score(y, p):
    return round(mean_absolute_error(y, p), 3), round(mean_squared_error(y, p) ** 0.5, 3)


def load_groups(cfg, root):
    path = cfg["find"](root)
    if not path:
        return None
    df = cfg["load"](path)
    groups = list(iter_groups(df, cfg["group"]))
    if cfg.get("min_coverage"):
        tmin = min(g.index.min() for _, g in groups)
        tmax = max(g.index.max() for _, g in groups)
        steps = len(pd.date_range(tmin, tmax, freq=cfg["freq"]))
        groups = [(k, g) for k, g in groups
                  if g["PM2.5"].notna().sum() / steps >= cfg["min_coverage"]]
    return groups


def evaluate_split(built, cfg, h, cutoff, end, forest, log, quiet=False, do_tune=False):
    """built = {group: (X, y, B)}. Returns a list of unit results for one split."""
    f = FREQ[cfg["freq"]]
    units = {}
    for k, (X, y, B) in built.items():
        Xtr, Xte, ytr, yte, bte = split(X, y, B, cutoff, h, f["step"], end)
        if len(Xtr) < MIN_TRAIN or len(Xte) < MIN_TEST:
            if not quiet:
                log(f"   skip {k} h={h}: train {len(Xtr)}, test {len(Xte)}")
            continue
        units[k] = (Xtr, Xte, ytr, yte, bte)
    if cfg.get("pooled") and units:
        units = {"pooled": tuple(pd.concat(p) for p in zip(*units.values()))}
    out = []
    for k, (Xtr, Xte, ytr, yte, bte) in units.items():
        preds = {"persistence": Xte["PM2.5_lag0"].values,
                 "seasonal_naive": bte["seasonal_naive"].values,
                 "trailing_mean": bte["trailing_mean"].values}
        for mname, m in make_models(forest).items():
            m.fit(Xtr, ytr)
            preds[mname] = m.predict(Xte)
        tuned = []
        if do_tune:
            jobs = [("ridge", ridge_model, RIDGE_GRID)]
            if forest:
                jobs.append(("forest", forest_model, FOREST_GRID))
            for base, factory, grid in jobs:
                search = (lambda **p: factory(n_estimators=TUNE_TREES, **p)) \
                    if base == "forest" else factory
                params, val = tune(search, grid, Xtr, ytr, h, f["step"])
                preds[f"{base}_tuned"] = factory(**params).fit(Xtr, ytr).predict(Xte)
                tuned.append(dict(group=k, model=f"{base}_tuned", params=str(params),
                                  val_MAE=round(val, 3)))
        out.append(dict(group=k, preds=preds, y=yte.values, time=Xte.index,
                        n_train=len(Xtr), n_test=len(Xte), tuned=tuned))
    return out


def to_rows(name, cfg, h, units, **extra):
    unit = FREQ[cfg["freq"]]["unit"]
    rows = []
    for u in units:
        for mname, p in u["preds"].items():
            mae, rmse = score(u["y"], p)
            rows.append(dict(dataset=name, group=u["group"], horizon=h, unit=unit,
                             model=mname, MAE=mae, RMSE=rmse,
                             n_train=u["n_train"], n_test=u["n_test"], **extra))
    return rows


def fold_windows(groups, cfg, K):
    """K consecutive test windows covering roughly the last year before the main
    cutoff plus the main test period (never earlier than 40% into the data)."""
    tmin = min(g.index.min() for _, g in groups)
    tmax = max(g.index.max() for _, g in groups)
    step = FREQ[cfg["freq"]]["step"]
    start = max(pd.Timestamp(cfg["cutoff"]) - pd.Timedelta(days=365),
                tmin + 0.4 * (tmax - tmin))
    width = (tmax + step - start) / K
    return [(start + i * width, start + (i + 1) * width) for i in range(K)]


def param_rows(name, h, units, fold):
    return [dict(dataset=name, horizon=h, fold=fold, **t)
            for u in units for t in u.get("tuned", [])]


def run_dataset(name, cfg, root, forest, log, rolling=0, do_tune=False):
    t0 = time.time()
    groups = load_groups(cfg, root)
    if groups is None:
        log(f"[{name}] skipped: file not found in {root}")
        return [], [], {}, []
    f = FREQ[cfg["freq"]]
    log(f"\n[{name}] {len(groups)} group(s), freq {cfg['freq']}, test starts {cfg['cutoff']}")
    rows, rrows, recs, prows = [], [], {}, []
    wins = fold_windows(groups, cfg, rolling) if rolling else []
    for h in f["horizons"]:
        built = {k: build(g, cfg, h) for k, g in groups}
        main = evaluate_split(built, cfg, h, cfg["cutoff"], None, forest, log,
                              do_tune=do_tune)
        rows += to_rows(name, cfg, h, main)
        prows += param_rows(name, h, main, 0)
        recs[h] = main
        for i, (c, e) in enumerate(wins, 1):
            fo = evaluate_split(built, cfg, h, c, e, forest, log, quiet=True,
                                do_tune=do_tune)
            rrows += to_rows(name, cfg, h, fo, fold=i, test_start=str(c)[:10],
                             test_end=str(e)[:10])
            prows += param_rows(name, h, fo, i)
        log(f"   horizon {h}{f['unit']} done  [{time.time() - t0:.0f}s]")
    return rows, rrows, recs, prows


# ---------- confidence intervals ----------
BLOCK = {"h": pd.Timedelta(days=7), "D": pd.Timedelta(days=14)}
PAIRS = [("ridge", "persistence"), ("forest", "persistence"),
         ("ridge", "trailing_mean"), ("forest", "trailing_mean"), ("forest", "ridge"),
         ("ridge_tuned", "persistence"), ("forest_tuned", "persistence"),
         ("ridge_tuned", "trailing_mean"), ("forest_tuned", "trailing_mean"),
         ("ridge_tuned", "ridge"), ("forest_tuned", "forest"),
         ("forest_tuned", "ridge_tuned")]
SHOW_PAIRS = [("ridge", "persistence"), ("ridge_tuned", "persistence"),
              ("forest", "persistence"), ("forest_tuned", "persistence"),
              ("ridge_tuned", "ridge"), ("forest_tuned", "forest"),
              ("forest_tuned", "ridge_tuned")]


def bootstrap_gains(units, freq, B=1000, seed=0):
    """Gain of model A over model B (= 100 * (1 - A/B)) for MAE and RMSE, as the mean
    over groups, with a block-bootstrap 95% interval. Time blocks are resampled jointly
    across groups so cross-station correlation is kept."""
    models = list(units[0]["preds"])
    t0 = min(u["time"].min() for u in units)
    blk = lambda t: np.asarray((t - t0) // BLOCK[freq], dtype=int)
    nb = max(blk(u["time"]).max() for u in units) + 1
    G = len(units)
    N = np.zeros((nb, G))
    S = {m: np.zeros((nb, G)) for m in models}
    S2 = {m: np.zeros((nb, G)) for m in models}
    for gi, u in enumerate(units):
        b = blk(u["time"])
        N[:, gi] = np.bincount(b, minlength=nb)
        for m in models:
            e = np.abs(u["y"] - u["preds"][m])
            S[m][:, gi] = np.bincount(b, weights=e, minlength=nb)
            S2[m][:, gi] = np.bincount(b, weights=e * e, minlength=nb)
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(nb)] + [np.bincount(rng.integers(0, nb, nb), minlength=nb)
                                   for _ in range(B)])          # row 0 = the real sample
    n = W @ N

    def metric(m, kind):
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "MAE":
                v = (W @ S[m]) / n
            else:
                v = np.sqrt((W @ S2[m]) / n)
        v = np.where(n > 0, v, np.nan)
        return np.nanmean(v, axis=1)

    res = []
    for a, bm in PAIRS:
        if a not in models or bm not in models:
            continue
        for kind in ("MAE", "RMSE"):
            with np.errstate(invalid="ignore", divide="ignore"):
                g = 100 * (1 - metric(a, kind) / metric(bm, kind))
            lo, hi = np.nanpercentile(g[1:], [2.5, 97.5])
            res.append(dict(model=a, baseline=bm, metric=kind, gain_pct=float(round(g[0], 2)),
                            ci_low=float(round(lo, 2)), ci_high=float(round(hi, 2)),
                            excludes_zero=bool(lo > 0 or hi < 0)))
    return res


def ci_table(name, cfg, recs, B):
    rows = []
    for h, units in recs.items():
        if not units:
            continue
        for r in bootstrap_gains(units, cfg["freq"], B=B):
            rows.append(dict(dataset=name, horizon=h,
                             unit=FREQ[cfg["freq"]]["unit"], **r))
    return rows


def summarize_ci(ci, log):
    fmt = lambda r: (f"{r.gain_pct:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]"
                     + ("" if r.excludes_zero else " ns"))
    for ds, g in ci.groupby("dataset", sort=False):
        log(f"\n=== {ds}: MAE gain % with 95% block-bootstrap interval "
            "(ns = interval includes 0) ===")
        out = []
        for h, gh in g[g.metric == "MAE"].groupby("horizon"):
            row = {"horizon": f"{h}{gh.unit.iloc[0]}"}
            for r in gh.itertuples():
                if (r.model, r.baseline) in SHOW_PAIRS:
                    row[f"{r.model}_vs_{r.baseline}"] = fmt(r)
            out.append(row)
        log(pd.DataFrame(out).to_string(index=False))


def summarize_rolling(rdf, log):
    for ds, g in rdf.groupby("dataset", sort=False):
        log(f"\n=== {ds}: rolling origin, MAE gain % over persistence, per fold ===")
        out = []
        for h, gh in g.groupby("horizon"):
            row = {"horizon": f"{h}{gh.unit.iloc[0]}"}
            for m in MODEL_ORDER:
                gains = []
                for fo, gf in gh.groupby("fold"):
                    mae = gf.pivot_table(index="group", columns="model", values="MAE").mean()
                    if m in mae.index and "persistence" in mae.index:
                        gains.append(100 * (1 - mae[m] / mae["persistence"]))
                if gains:
                    row[f"{m}_by_fold"] = " ".join(f"{x:+.0f}" for x in gains)
                    row[f"{m}_folds_won"] = f"{sum(x > 0 for x in gains)}/{len(gains)}"
            out.append(row)
        log(pd.DataFrame(out).to_string(index=False))


MODEL_ORDER = ["ridge", "ridge_tuned", "forest", "forest_tuned"]


def summarize(res, log):
    for ds, g in res.groupby("dataset", sort=False):
        log(f"\n=== {ds} ===  (mean MAE over groups; _vs_pers = % lower than persistence; "
            "_beats = groups where the model beats persistence)")
        out = []
        for h, gh in g.groupby("horizon"):
            mae = gh.pivot_table(index="group", columns="model", values="MAE")
            m = mae.mean()
            row = {"horizon": f"{h}{gh.unit.iloc[0]}", "n_groups": len(mae)}
            for c in ["persistence", "trailing_mean"] + MODEL_ORDER:
                if c in m:
                    row[c] = round(m[c], 2)
            for c in MODEL_ORDER:
                if c in m:
                    row[f"{c}_vs_pers_%"] = round((1 - m[c] / m["persistence"]) * 100, 1)
            for c in MODEL_ORDER:
                if c in m:
                    row[f"{c}_beats"] = f"{int((mae[c] < mae['persistence']).sum())}/{len(mae)}"
            out.append(row)
        log(pd.DataFrame(out).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--no-forest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rolling", type=int, default=0, help="number of rolling-origin folds")
    ap.add_argument("--no-ci", action="store_true", help="skip bootstrap intervals")
    ap.add_argument("--tune", action="store_true",
                    help="also fit tuned models (inner time-series CV on the training data)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--out", default="multibench_results.csv")
    a = ap.parse_args()
    reg = registry()
    if a.list:
        for k, c in reg.items():
            print(f"{k:<18} freq {c['freq']}  test from {c['cutoff']}")
        return
    names = list(reg) if a.datasets == "all" else a.datasets.split(",")
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    rows, rrows, cirows, prows = [], [], [], []
    for n in names:
        if n not in reg:
            raise SystemExit(f"unknown dataset {n}; use --list")
        try:
            r, rr, recs, pr = run_dataset(n, reg[n], a.root, not a.no_forest, log,
                                          a.rolling, a.tune)
            rows += r
            rrows += rr
            prows += pr
            if not a.no_ci and recs:
                cirows += ci_table(n, reg[n], recs, a.boot)
        except Exception as e:                       # keep going if one dataset fails
            log(f"[{n}] FAILED: {type(e).__name__}: {e}")
    if not rows:
        raise SystemExit("no results")
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    summarize(res, log)
    if cirows:
        ci = pd.DataFrame(cirows)
        ci.to_csv("multibench_ci.csv", index=False)
        summarize_ci(ci, log)
    if rrows:
        rdf = pd.DataFrame(rrows)
        rdf.to_csv("multibench_rolling.csv", index=False)
        summarize_rolling(rdf, log)
    if prows:
        pd.DataFrame(prows).to_csv("multibench_tuned_params.csv", index=False)
    log(f"\nSaved {a.out} ({len(res)} rows)"
        + (", multibench_tuned_params.csv" if prows else "")
        + (", multibench_ci.csv" if cirows else "")
        + (", multibench_rolling.csv" if rrows else ""))
    with open("multibench_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
