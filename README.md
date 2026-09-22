# Non-Invasive Cardiovascular Clinical Diagnostic Pipeline with Explainable AI

Binary classification pipeline for angiographically confirmed coronary artery stenosis (>= 50% vessel narrowing) using the UCI Cleveland Heart Disease dataset. Optimized for >= 90% diagnostic sensitivity, calibrated risk probabilities, and per-patient SHAP explanations.

## Dataset

- **Source:** UCI Machine Learning Repository, Cleveland subset (303 patients, 13 features)
- **Target:** `0` = < 50% stenosis, `1` = >= 50% stenosis (binarized from raw values 0-4)
- **Class balance:** 138 normal (45.5%) / 165 diseased (54.5%)
- **Missing values:** 6 total (4 in `ca`, 2 in `thal`), median/mode imputed within the pipeline

## Setup

The project uses a managed virtual environment. With `uv`:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Run the full pipeline:

```bash
PYTHONPATH=. .venv/bin/python -m src.train      # train + calibrate + save artifact
PYTHONPATH=. .venv/bin/python -m src.evaluate   # generate evaluation figures + metrics
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

Launch the clinical dashboard:

```bash
uv run app/app.py
# or
.venv/bin/streamlit run app/app.py
```

## Architecture

```
src/config.py         paths, column groups, thresholds
src/data_loader.py    download + parse raw Cleveland data
src/preprocessor.py   clinical feature engineering + ColumnTransformer
src/models.py         candidate model factories (LR / RF / XGB)
src/calibration.py    PlattCalibrator (cross-validated sigmoid scaling)
src/train.py          Stratified 5-fold CV, model selection, threshold tuning
src/evaluate.py       ROC / PR / calibration / decision curve figures
src/explainer.py      SHAP explanations + counterfactual simulator
app/app.py            Streamlit clinical dashboard
tests/test_pipeline.py
```

The trained artifact (`artifacts/best_model.joblib`) stores four separate objects so that calibration and SHAP can each operate on the correct input:

- `transform_pipeline` — feature engineering + preprocessing (raw -> 24 features)
- `base_model` — fitted estimator, used for SHAP explanations
- `calibrated_model` — Platt-scaled probabilities
- `threshold`, `threshold_metrics`, `metrics`, `feature_names`

## Results

Stratified 5-fold out-of-fold cross-validation on all 303 patients:

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.9102 | 0.9025 | 0.791 | 0.902 | 0.1160 |
| Random Forest | 0.9052 | 0.9006 | 0.755 | 0.896 | 0.1258 |
| XGBoost | 0.8985 | 0.8924 | 0.799 | 0.854 | 0.1307 |

**Selected model:** logistic regression (elastic-net penalty, `saga` solver).

Held-out test set (n=61, stratified 80/20 split, threshold 0.319):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9686 |
| PR-AUC | 0.9573 |
| Sensitivity | 0.9643 |
| Specificity | 0.7576 |
| Brier (calibrated) | 0.0727 |
| Brier (uncalibrated) | 0.0713 |

The decision threshold is not 0.5. It is tuned by sweeping probabilities and selecting the highest-specificity point that still satisfies >= 90% sensitivity, which is the clinically appropriate asymmetry (a missed stenosis is far costlier than a false alarm).

## Explainability

SHAP values are computed on the *uncalibrated* base model via `LinearExplainer` (or `TreeExplainer` if a tree model is selected). Calibration is a monotonic rescaling of the score, so it changes the probability but not the ranking or direction of feature contributions. The additivity property is asserted in the test suite:

```
base_value + sum(shap_values) == decision_function(X)
```

## Counterfactual simulator

The dashboard's What-If tab locks immutable history (age, sex, prior vessel damage, ECG findings) and varies only modifiable factors (resting blood pressure, cholesterol, peak heart rate). The artifact's `transform_pipeline` recomputes the engineered features (`rpp`, `hr_reserve`) for each simulated patient so the counterfactual is internally consistent.

## Caveats

These are known limitations, not defects:

1. **Calibration did not demonstrably improve the Brier score on this test split.** Calibrated 0.0727 vs uncalibrated 0.0713. On n=61 this difference is within noise. Calibration is retained because Platt scaling is standard practice for tree/linear score outputs and the reliability diagram is the more informative evidence, but the pipeline does not claim a measured improvement.
2. **The Platt calibrator saturates near 1.0.** Very-high-risk outputs should be read as "very high risk," not as precise probabilities. Counterfactual deltas for patients already above ~0.95 are compressed.
3. **Mild model-selection leakage.** Cross-validation metrics used for model selection were computed over all 303 rows before the 80/20 test split. For a stricter protocol, selection should be performed on the training split only.
4. **Small dataset.** 303 patients with 6 missing values. Confidence intervals on held-out metrics are wide.
5. **`CalibratedClassifierCV` was not used.** On scikit-learn 1.9.1 / Python 3.14 it returned constant probabilities for varying inputs. `src/calibration.py` implements the equivalent cross-validated Platt scaling explicitly.

## Environment notes

- Python 3.14, scikit-learn 1.9.1
- `xgboost` pinned to `< 2.1`; version 3.x pulls a 305 MB CUDA/NCCL wheel that is unnecessary for a 303-row CPU workload.
- This is a research/educational pipeline, not a certified medical device. Do not use for clinical decision-making.
