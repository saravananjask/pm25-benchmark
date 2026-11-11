# Data

All six benchmarked datasets are third-party open data. They are **not** stored in this
repository. Download each one, put the files in one folder, and pass that folder to the
scripts. File and folder names matter: the loaders look for the patterns below.

| Key | The loader looks for | Source (confirm before release) | Licence (confirm) |
| --- | --- | --- | --- |
| beijing_multisite | `beijing+multi*.zip` | UCI Machine Learning Repository, Beijing Multi-Site Air-Quality Data | [confirm] |
| beijing_pm25 | `beijing+pm2*.zip` | UCI Machine Learning Repository, Beijing PM2.5 Data (id 381) | [confirm] |
| five_cities | folder `pm2+5+data+of+five+chinese+cities` | UCI Machine Learning Repository, PM2.5 Data of Five Chinese Cities (id 394) | [confirm] |
| india_city_day | `archive.zip` (contains `city_day.csv`) | Kaggle, air quality data in India (CPCB), city_day table | [confirm] |
| epa_daily | `daily_88101_*.zip` | US EPA Air Quality System, pre-generated daily PM2.5 file (parameter 88101); the paper uses 2023 | US government data [confirm] |
| openaq | folder `data` of `*.csv.gz` | OpenAQ archive, location 2178 | [confirm] |

Three more loaders exist but are not part of the benchmark: Italy (UCI id 360, no PM2.5
column), EEA annual averages (`*.parquet`), NOAA GHCN-Daily (`*.dly`).

## Checksums

Run `python scripts/checksums.py "<data folder>"` and paste the table below, so that others
can check they hold the same files.

(paste the checksum table here)
