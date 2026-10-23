# Garmin Data Exploration

A single-subject study of my own wearable data: how sleep, overnight HRV, resting heart rate and stress relate to the menstrual cycle, and how they changed through a marathon training block.

**[Read the write-up →](portfolio/sleep-study.md)** · [interactive version](portfolio/sleep-study.html) (download and open in a browser)

## Headline results

- **The menstrual cycle lowers HRV, not the other way round.** HRV drops ~7 ms below my baseline in the 5 days before a period and overnight resting HR rises ~3.4 bpm (Mann–Whitney p < 10⁻⁹). Follicular-phase HRV and stress do not predict cycle length (ρ ≈ 0, 14 cycles).
- **HRV tracks sleep quality far more than sleep length** (ρ = 0.61 vs 0.18, 489 nights).
- **Daytime stress costs REM and total sleep, not deep sleep.** After the most stressed fifth of days I sleep ~65 min less than after the calmest fifth.
- **Marathon block:** running HR fell (169 → 163 bpm) while resting HR rose and HRV fell (64 → 56 ms), a fatigue signature. Higher-HRV weeks were faster weeks (ρ = −0.35, p = 0.027).
- **Prediction:** a regression on information available the day before explains ~44% of the variance in tonight's HRV and ~42% in sleep score on held-out nights.

## Pipeline

```
Garmin GDPR export (JSON + raw FIT)  ─┐
Clover period-tracker export          ├─> etl.py (PySpark) ─> daily / runs / cycles tables ─> report.py ─> HTML + Markdown report
extract_fit_hrv.py (FIT → HRV csv)   ─┘
```

| File | What it does |
|---|---|
| `extract_fit_hrv.py` | Pulls nightly HRV (`hrv_status_summary` messages) from the watch's raw FIT files. These cover May–Sep 2025, which Garmin's Health Status JSON is missing. |
| `etl.py` | PySpark job: parses and flattens the Garmin JSON (sleep, health status, daily summaries, activities), decodes the Clover export, assigns cycle phases, and joins everything into one row per day and one row per run. |
| `report.py` | Statistics (Spearman with Fisher-z CIs, Kruskal–Wallis, Mann–Whitney, OLS with Newey–West errors and a time-ordered hold-out) and the Plotly dashboard. `--public` builds the portfolio version with personal dates removed, plus a Markdown file with static PNG figures. |
| `scan_fit.py` | Utility: inventories FIT file types and HRV-related messages. |

## Running it

Requires Python 3.11+ and Java 17 or 21 (for Spark).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export JAVA_HOME=$(/usr/libexec/java_home -v 21)   # macOS

# 1. HRV from raw FIT files (unzip DI_CONNECT/DI-Connect-Uploaded-Files/*.zip first)
.venv/bin/python extract_fit_hrv.py <unzipped fit dir> data/fit_hrv_raw.csv
# 2. Spark ETL
.venv/bin/python etl.py <export>/DI_CONNECT <clover export file> data data/fit_hrv_raw.csv
# 3. Reports
.venv/bin/python report.py data report.html                                   # full, private
.venv/bin/python report.py --public data portfolio/sleep-study.html           # portfolio-safe + .md
```

Training dates (base build start, race day) are set at the top of `etl.py` and `report.py`.

## Data

The raw exports and derived tables are **not** in this repository: they're personal health data. `data/` is git-ignored apart from the public findings text.

## Limitations

n = 1 and observational, so these are correlations, not causal effects. Ovulation is estimated (next period − 14 days), not measured. Garmin sleep staging and "stress" are proprietary estimates, and Garmin stress is itself derived from HRV.
