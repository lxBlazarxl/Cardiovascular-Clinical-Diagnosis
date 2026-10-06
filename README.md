# Non-Invasive Cardiovascular Risk Stratification

Two deliverables, nothing else:

| File | Purpose |
|---|---|
| `honors_mp.ipynb` | EDA + training + calibration + threshold tuning + model export |
| `index.html` | Self-contained clinical dashboard (open by double-click, no server) |

Supporting: `data/raw/cleveland.csv` (input), `model_export.json` (generated payload),
`EXPLANATIONS.md` (full walkthrough), `OVERHAUL.md` (restructure record).

> **New here?** Read [`EXPLANATIONS.md`](EXPLANATIONS.md). It explains every column, every
> notebook step, every dashboard control, how to run and verify everything, and — importantly —
> what this model cannot do.

## Dataset

- UCI Cleveland Heart Disease, 303 patients, 13 features, headerless CSV with `?` for missing.
- Target binarized: `0` = no stenosis, `1` = stenosis (raw severity > 0).
- Class balance: 164 no disease / 139 disease (positive rate 0.459).

## Notebook

Run all cells in `honors_mp.ipynb`. It:

1. Loads and cleans the raw CSV (`?` -> NaN).
2. Performs EDA: shape, dtypes, missingness, class balance, distributions, correlations.
3. Engineers `rpp = trestbps * thalach / 100` and `hr_reserve = thalach / (220 - age)`.
4. Builds a `ColumnTransformer` (median impute + scale for numeric, mode impute + one-hot for categorical, mode impute + scale for binary).
5. Trains a single logistic regression.
6. Fits Platt (sigmoid) calibration on out-of-fold decision scores.
7. Tunes the decision threshold on the **training** split to hold sensitivity >= 0.90, tie-breaking on max specificity.
8. Reports test metrics and writes `model_export.json`.

Reference results on the held-out split (n=61):

| Metric | Value |
|---|---|
| ROC-AUC | 0.958 |
| PR-AUC | 0.941 |
| Sensitivity | 0.964 |
| Specificity | 0.727 |
| Threshold | 0.355 |

## Dashboard

Open `index.html` directly (`file://`). It inlines the CSS, the inference JS, and the full model payload — no network access required beyond the Google Fonts CDN link for `Inter` / `IBM Plex Mono`.

Enter patient values, submit, and read the calibrated risk, verdict against the tuned threshold, and per-feature coefficient contributions. `High-risk` and `Low-risk` sample buttons load preset patients.

The browser reproduces the sklearn pipeline exactly: it recomputes `rpp` / `hr_reserve`, applies the exported scaler statistics, one-hot expands categoricals, dots with the exported coefficients, then applies the Platt calibration. Sanity patient matches the notebook at p = 0.996349.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/python -m jupyter nbconvert --to notebook --execute honors_mp.ipynb  # optional
```

To regenerate the payload without Jupyter, execute the notebook's code cells in plain Python with the `Agg` matplotlib backend.

## Caveats

- The threshold is not 0.5; it is chosen to favor sensitivity, because a missed stenosis is costlier than a false alarm.
- The threshold is tuned on training data only, never on the test split, so the reported test metrics are an honest held-out estimate. An earlier version of this project picked the threshold on the test set, which reported specificity 0.818 instead of 0.727 — that number was optimistic and is not used.
- Platt scaling is retained; on this small test split it did not measurably improve Brier score.
- 303 patients with a few missing values: confidence intervals are wide. Research prototype only, not for clinical use.
