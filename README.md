# pm25-benchmark

A leakage-safe benchmark toolkit for short-term PM2.5 forecasting on open datasets.
It cleans six open datasets, builds forecasting tables without using future information,
and compares ridge regression and a random forest against three baselines, with
block-bootstrap 95% intervals, rolling-origin folds and nested time-ordered tuning.

## Install

Tested on Python 3.14.7. Python 3.10 or newer should work.

    python -m pip install -r requirements.txt

## Data

See `DATA.md`. Put all datasets in one folder (called `DATA` below).

## Run

Check the data loads (writes `audit_report.txt` in the current folder):

    python audit.py DATA

Quick run, ridge and baselines only (a few minutes):

    python multibench.py DATA --no-forest

Full run that produced the paper's tables (about 100 minutes on a laptop):

    python multibench.py DATA --tune --rolling 4

Other options: `--datasets epa_daily,openaq`, `--no-ci`, `--boot 1000`, `--list`.

## Outputs

| File | Content |
| --- | --- |
| `multibench_results.csv` | MAE and RMSE for every dataset, group, horizon and model |
| `multibench_ci.csv` | gains over baselines with 95% block-bootstrap intervals |
| `multibench_rolling.csv` | rolling-origin fold results (with `--rolling`) |
| `multibench_tuned_params.csv` | settings chosen by tuning (with `--tune`) |
| `multibench_report.txt` | the printed summary tables |

The files from the paper's run are in `results/`. The forest and the bootstrap use fixed
seeds, so a rerun with the pinned versions in `requirements.txt` on the same data should
reproduce them. Compare your output with `results/` after a full run.

## Protocol in one paragraph

The target is PM2.5 at t + h and features use information up to t only. Feature gaps are
carried forward for at most 3 steps (hourly) or 2 (daily); the target is never filled.
Each dataset has one fixed test start date. A training row is used only if its target time
is before the test start (purge gap). Imputation and scaling are fitted on training rows
only. Baselines: persistence, seasonal naive, trailing mean.

## Tests

    python -m pytest

The tests use small synthetic data and need no downloads.

## Known limitations

The forest's minimum leaf size sat at the upper edge of its search grid
(`FOREST_GRID` in `multibench.py`) and was not widened for the paper's results.

## Licence and citation

MIT, see `LICENSE`. Citation details are in `CITATION.cff`.
