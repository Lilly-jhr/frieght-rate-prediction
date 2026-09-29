# Freight Rate Prediction Challenge

> **Note:** this is the assessment's originally-provided `readme.md`, preserved here under a
> new filename. Windows treats `readme.md` and `README.md` as the same file, so it could not
> coexist with this repository's own `README.md`. Its content is unchanged except for the
> filename corrections described at the bottom.

See `freight-rate-ml-assessment.md` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train-test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation-predictions-template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december-chart-inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions outputs/december-chart-inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

---

## Corrections applied to this file

The original referenced filenames that do not exist on disk, so its `score.py` command failed
as written:

| Original reference | Actual file |
|---|---|
| `freight-rate-ml-assessment.pdf` | `freight-rate-ml-assessment.md` |
| `data/train_test.csv` | `data/train-test.csv` |
| `data/validation_predictions_template.csv` | `data/validation-predictions-template.csv` |
| `data/december_chart_inputs.csv` | `data/december-chart-inputs.csv` |

The `--december-predictions` path was also repointed to `outputs/december-chart-inputs.csv`,
where this solution writes the filled file (leaving the provided `data/` copy untouched).
