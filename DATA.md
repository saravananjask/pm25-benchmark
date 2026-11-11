# Data

All six benchmarked datasets are third-party open data. They are **not** stored in this
repository. Download each one, put the files in one folder, and pass that folder to the
scripts. File and folder names matter: the loaders look for the patterns below.
Please cite the original data sources when you use them.

| Key | The loader looks for | Source | Licence |
| --- | --- | --- | --- |
| beijing_multisite | `beijing+multi*.zip` | UCI Machine Learning Repository, Beijing Multi-Site Air Quality, Chen S. (2017), https://doi.org/10.24432/C5RK5G (see Zhang et al. 2017, Proc. R. Soc. A 473:20170457) | CC BY 4.0 |
| beijing_pm25 | `beijing+pm2*.zip` | UCI Machine Learning Repository, Beijing PM2.5, Chen S. (2015), https://doi.org/10.24432/C5JS49 (see Liang et al. 2015, Proc. R. Soc. A 471:20150257) | CC BY 4.0 |
| five_cities | folder `pm2+5+data+of+five+chinese+cities` | UCI Machine Learning Repository, PM2.5 Data of Five Chinese Cities, Chen S. (2016), https://doi.org/10.24432/C52K58 (see Liang et al. 2016, J. Geophys. Res. Atmos.) | CC BY 4.0 |
| india_city_day | `archive.zip` (contains `city_day.csv`) | Kaggle, "Air Quality Data in India (2015 - 2020)", compiled from Central Pollution Control Board (CPCB) data | Check the dataset page on Kaggle and the CPCB terms before reuse |
| epa_daily | `daily_88101_*.zip` | US EPA Air Quality System (AQS), AirData pre-generated daily PM2.5 file (parameter 88101); the benchmark uses 2023 | US EPA open data; see the EPA AirData page for its terms of use |
| openaq | folder `data` of `*.csv.gz` | OpenAQ archive, location 2178 | Varies by data provider; see https://docs.openaq.org/docs/data-policy |

Three more loaders exist but are not part of the benchmark: Italy (UCI id 360, no PM2.5
column), EEA annual averages (`*.parquet`), NOAA GHCN-Daily (`*.dly`).

## Checksums

Run `python scripts/checksums.py "<data folder>"` and compare with the table below, so that
others can check they hold the same files.

| dataset | file or folder | SHA-256 |
| --- | --- | --- |
| beijing_multisite | beijing+multi+site+air+quality+data.zip | b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4 |
| beijing_pm25 | beijing+pm2+5+data.zip | 40cdd29c24918e855133dce280ff59779bb0b5852898e05bafcf422a4cc0a175 |
| five_cities | pm2+5+data+of+five+chinese+cities | 892ddf419757b933993b479395a02433539e193017d2723db191e87a3ddaa2c2 |
| india_city_day | archive.zip | 4ce4e048e6698b277b6d8f14d830c1f7434769416302dcdbab4f0826ee750ff9 |
| epa_daily | daily_88101_2023.zip | 4b40502e8147419643a1120c666c4aa952633094f21f65475844172bcc78833b |
| openaq | data | 1909648a69665163bc8b7d2f3732b7d257c4a1a3b5ede5dc2d7025ae8deaeac5 |

Folder checksums (five_cities, openaq) hash every file's relative path and SHA-256, sorted.

The final-run result file `results/multibench_results.csv` has SHA-256
`58F3F38B515FAB6593050C0AEB5EF7D780DF089D1521F5A917989CB0A050E4B2`.
