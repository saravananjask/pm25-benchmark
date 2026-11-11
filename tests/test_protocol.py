"""Tests on small synthetic data: no downloads needed. Run:  python -m pytest"""
import numpy as np
import pandas as pd

import loaders as L
import multibench as M

STEP_H = pd.Timedelta(hours=1)


def hourly(n, seed=0, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq="h")
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.randn()
    seasonal = 5 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.DataFrame({"PM2.5": 50 + 10 * x + seasonal}, index=idx)


def test_features_and_target_line_up():
    idx = pd.date_range("2020-01-01", periods=300, freq="h")
    g = pd.DataFrame({"PM2.5": np.arange(300, dtype=float)}, index=idx)
    cfg = dict(freq="h", features=["PM2.5"])
    X, y, B = M.build(g, cfg, 6)
    assert ((y - X["PM2.5_lag0"]) == 6).all()          # target is 6 steps ahead


def test_changing_the_future_does_not_change_past_features():
    g = hourly(400)
    cfg = dict(freq="h", features=["PM2.5"])
    X1, _, _ = M.build(g, cfg, 6)
    g2 = g.copy()
    g2.iloc[200:] += 1000.0
    X2, _, _ = M.build(g2, cfg, 6)
    past = X1.index[X1.index < g.index[200]]
    pd.testing.assert_frame_equal(X1.loc[past], X2.loc[past])


def test_gap_filling_is_causal_and_never_touches_the_target():
    g = hourly(400)
    g["PM10"] = g["PM2.5"] * 2
    g.iloc[100:106, g.columns.get_loc("PM10")] = np.nan      # 6 missing, limit is 3
    g.iloc[150:155, g.columns.get_loc("PM2.5")] = np.nan     # missing target values
    cfg = dict(freq="h", features=["PM2.5", "PM10"])
    X, y, _ = M.build(g, cfg, 6)
    t = g.index
    assert X.loc[t[100:103], "PM10_lag0"].notna().all()      # filled forward, up to 3
    assert X.loc[t[103]:t[105], "PM10_lag0"].isna().all()    # longer gap stays missing
    assert X.loc[t[100], "PM10_lag0"] == g["PM10"].iloc[99]  # value carried from the past
    for pos in range(144, 149):                              # target would fall in 150..154
        assert t[pos] not in y.index


def test_purge_gap_keeps_training_targets_before_the_cutoff():
    g = hourly(3000)
    cfg = dict(freq="h", features=["PM2.5"])
    cutoff = g.index[2000]
    for h in (1, 6, 24, 48):
        X, y, B = M.build(g, cfg, h)
        Xtr, Xte, ytr, yte, bte = M.split(X, y, B, cutoff, h, STEP_H)
        assert (Xtr.index + h * STEP_H).max() < cutoff
        assert Xte.index.min() >= cutoff
        assert len(Xtr) > 0 and len(Xte) > 0


def test_inner_folds_are_ordered_and_purged():
    times = pd.date_range("2020-01-01", periods=2000, freq="h")
    for h in (1, 24):
        folds = list(M.inner_folds(times, h, STEP_H))
        assert len(folds) == M.INNER_FOLDS
        last_val_end = None
        for tr, va in folds:
            assert len(tr) > 0 and len(va) > 0
            assert times[tr].max() + (h + 1) * STEP_H <= times[va].min()
            if last_val_end is not None:
                assert times[va].min() >= last_val_end
            last_val_end = times[va].max()


def _units(seed=0):
    rng = np.random.RandomState(seed)
    units = []
    for g in range(2):
        t = pd.date_range("2020-01-01", periods=24 * 120, freq="h")
        y = 50 + 10 * rng.randn(len(t))
        units.append(dict(group=g, y=y, time=t, preds={
            "persistence": y + 10 * rng.randn(len(t)),
            "ridge": y + 3 * rng.randn(len(t))}))
    return units


def test_bootstrap_is_reproducible_and_detects_a_real_gain():
    a = M.bootstrap_gains(_units(), "h", B=200, seed=0)
    b = M.bootstrap_gains(_units(), "h", B=200, seed=0)
    assert a == b
    row = next(r for r in a if r["model"] == "ridge" and r["baseline"] == "persistence"
               and r["metric"] == "MAE")
    assert row["gain_pct"] > 0 and row["ci_low"] > 0


def test_range_limits_set_values_to_missing_and_count_them():
    df = pd.DataFrame({"PM2.5": [10.0, -5.0, 2000.0], "HUMI": [50.0, 120.0, 30.0]})
    out = L._clean(df)
    assert out["PM2.5"].isna().tolist() == [False, True, True]
    assert out.attrs["removed"] == {"PM2.5": 2, "HUMI": 1}


def test_daily_weather_is_attached_one_day_late():
    t = pd.date_range("2020-01-01", periods=72, freq="h")
    df = pd.DataFrame({"PM2.5": np.arange(72.0)}, index=t)
    daily = pd.DataFrame({"TAVG": [1.0, 2.0, 3.0]},
                         index=pd.date_range("2020-01-01", periods=3, freq="D"))
    out = L.add_previous_day(df, daily)
    assert out.loc["2020-01-02 12:00", "noaa_TAVG"] == 1.0    # day 2 sees day 1
    assert out.loc["2020-01-03 12:00", "noaa_TAVG"] == 2.0
    assert np.isnan(out.loc["2020-01-01 12:00", "noaa_TAVG"])  # day 1 sees nothing


def test_end_to_end_small_run_with_tuning():
    g = hourly(2400)
    cfg = dict(freq="h", features=["PM2.5"], pooled=False)
    cutoff = g.index[1800]
    built = {"A": M.build(g, cfg, 6)}
    units = M.evaluate_split(built, cfg, 6, cutoff, None, False, lambda s: None,
                             do_tune=True)
    assert len(units) == 1
    u = units[0]
    expected = {"persistence", "seasonal_naive", "trailing_mean", "ridge", "ridge_tuned"}
    assert expected <= set(u["preds"])
    assert u["time"].min() >= cutoff
    for name, p in u["preds"].items():
        assert len(p) == u["n_test"] and np.isfinite(p).all(), name
    assert u["tuned"] and "alpha" in u["tuned"][0]["params"]
