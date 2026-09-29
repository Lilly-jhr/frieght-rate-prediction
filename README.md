# Freight Rate Prediction — Solution

Predicts `posted_rate` for truckload freight shipments. Trains on 48,000 labeled loads
(Jan–Oct 2025) and predicts 12,000 validation loads (Nov–Dec 2025), plus a fixed-lane
December chart.

**Everything lives in one notebook:** [`freight_rate_prediction.ipynb`](freight_rate_prediction.ipynb).

---

## Data

The assessment dataset is **not committed** — this repository is public, and the data belongs to
the assessment. To reproduce, place the four provided CSVs in `data/`:

```
data/train-test.csv                       48,000 rows · Jan 1 – Oct 31 2025 · labeled
data/validation.csv                       12,000 rows · Nov 1 – Dec 31 2025 · unlabeled
data/validation-predictions-template.csv  12,000 rows · load_id,predicted_rate
data/december-chart-inputs.csv                 31 rows · Dec 1–31 2025
```

The generated outputs (`validation_predictions.csv`, `outputs/`, `reports/figures/`,
`scorer_results/`) are committed, so the results can be reviewed without re-running anything.

## Quick start

```bash
python -m pip install -r requirements.txt
jupyter notebook freight_rate_prediction.ipynb     # Kernel → Restart & Run All
```

Or run it headless:

```bash
python -m nbconvert --to notebook --execute --inplace freight_rate_prediction.ipynb
```

The notebook writes both deliverables and then invokes the official validator itself. To run
that validation step by hand:

```bash
python score.py --predictions validation_predictions.csv \
                --december-predictions outputs/december-chart-inputs.csv
```

## Outputs

| Path | What it is |
|---|---|
| `validation_predictions.csv` | 12,000 predictions — the graded submission |
| `outputs/december-chart-inputs.csv` | The 31 December rows with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | The chart produced by `score.py` |
| `reports/figures/*.png` | All 13 notebook figures, for the report |
| `reports/report.md` | Written report (convert to PDF/DOCX to submit) |

## Approach in brief

| | |
|---|---|
| **Model** | LightGBM (L1 loss, 4 seasonal harmonics), target `log($/mile)` reconstructed as `exp(pred) × distance` |
| **Validation** | 4-fold rolling-origin (walk-forward). Random K-fold rejected — it leaks the future into the past |
| **Out-of-time MAE** | **$98.38** vs **$192.12** for a lane-median baseline (49% better), on every fold |
| **Out-of-time MAPE** | 4.30% |

### Three findings that shaped the solution

1. **December lies outside the training day-of-year range** (335–365 vs 1–304). Because trees
   cannot extrapolate, raw date features yield a chart with only **7 distinct values across 31
   days** — seasonality silently replaced by a weekday cycle. Cyclical `sin/cos` encoding fixes
   it by placing December next to January, which the model *has* seen. See notebook §6.
2. **Two quick heuristics were wrong in opposite directions.** `quote_signal` ranks #1 by
   LightGBM gain but makes held-out error *worse* ($101.76 → $120.14). Meanwhile the variance
   decomposition rated day-of-week last, yet removing it costs $6.48 of MAE. Only out-of-time
   ablation got both right. See §8.
3. **Data-quality defects are load-bearing.** 672 corrupted labels make untrimmed L2 loss
   collapse to 200+ MAE, and render RMSE useless for model selection (it is ~625 for every good
   configuration). See §3 and §9.

### Notable choices

- **Geography is encoded as lat/lon, never as city or lane IDs.** 8 cities and 736 lanes appear
  only in validation; a one-hot column has no value to assign them, while coordinates place an
  unseen Charlotte between known Greensboro and Columbia.
- **Outliers are flagged, not deleted.** The holdout contains them too, so dropping them hides
  error rather than removing it.
- **One model serves both deliverables**, using only features computable from the December
  file's seven columns — so the chart cannot diverge from the predictions.
- **No ensemble.** A LightGBM/XGBoost blend was tested and rejected: residual correlation ≈1.0,
  because both models fail identically on the corrupted labels.
- **Also tried and rejected:** geographic cluster target encoding (worse), recency weighting (no
  effect), monotonic distance constraints (incompatible with L1). All recorded in §9.6.

## Repository layout

```
freight_rate_prediction.ipynb   the entire solution
score.py                        assessment-provided validator (unmodified)
assessment-readme.md            the assessment's original readme (renamed, see Notes)
requirements.txt
data/                           provided inputs (unmodified)
outputs/                        generated December file
reports/                        report + figures
validation_predictions.csv      generated submission
```

## Notes

- Seeded (`SEED = 42`) and deterministic; re-running reproduces identical outputs.
- `data/` is never modified — the filled December file is written to `outputs/`.
- The assessment's own instructions file was named `readme.md`. Windows treats that as the
  same file as this `README.md`, so the two cannot coexist — the original is preserved verbatim
  as [`assessment-readme.md`](assessment-readme.md).
- That file's filename references used underscores (`data/train_test.csv`) while the provided
  files use hyphens (`data/train-test.csv`), so its documented `score.py` command failed as
  written. Those paths have been corrected; the corrections are listed at the bottom of it.
