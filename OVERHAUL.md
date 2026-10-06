# OVERHAUL.md — Clinical ML Pipeline: Consolidation (COMPLETE)

> **Mission:** Condense the fractured repository into a lean footprint: a single EDA+training notebook, a single self-contained HTML dashboard, and only small necessary files. Core clinical logic (preprocessing, Platt calibration, sensitivity >= 90% thresholding, attribution) preserved.

**Status: COMPLETE.** All phases and acceptance gates satisfied.

---

## 1. Final End-State (achieved)

```
Honors_MP/
├── data/raw/cleveland.csv     # dataset input (303 rows, headerless)
├── honors_mp.ipynb            # Deliverable A — EDA + training + calibration + threshold + export
├── index.html                 # Deliverable B — self-contained dashboard (inline CSS/JS/payload)
├── model_export.json          # generated payload consumed/inlined by index.html
├── README.md                  # two-file usage docs
├── requirements.txt           # 6 deps
├── OVERHAUL.md                # this file
└── .gitignore
```

**Purged:** `app/` (Streamlit + scratch `_t*.py` + legacy dark theme), `public/` (source split out and inlined), `src/` (logic folded into notebook), `artifacts/` (figures + stale payloads), `tests/`, `_export.py`, `PROJECT_ROADMAP.md`, `.planning/`, `__pycache__`, `.pytest_cache`, `.trash`.

---

## 2. Confirmed Decisions

1. **Threshold rule** — hard `sensitivity >= 0.90`, tie-break on **max specificity**. Threshold tuned on the **training** split only (never the test set). Result: threshold `0.355`, sensitivity `0.964`, specificity `0.727`.
2. **Calibration** — Platt/sigmoid retained (payload shape `calibrator_coef` / `calibrator_intercept`). Isotonic rejected.
3. **Dataset** — `data/raw/cleveland.csv`, headerless, `?` -> NaN, 303 rows, target binarized `> 0`.
4. **Scope** — EDA + training in ONE notebook; plain logistic regression only (no XGBoost/tuning/ensembling/feature selection).

---

## 3. Delivered

### Notebook (`honors_mp.ipynb`)
- [x] Load/clean raw CSV; dtype + missingness + class-balance EDA
- [x] Feature engineering: `rpp = trestbps*thalach/100`, `hr_reserve = thalach/(220-age)`
- [x] `ColumnTransformer`: median+scale (num), mode+one-hot (cat), mode+scale (bin)
- [x] Logistic regression (C=1.0, max_iter=2000, lbfgs)
- [x] Platt calibration on out-of-fold decision scores
- [x] Threshold sweep holding sens >= 0.90, max specificity tie-break
- [x] Test metrics + plots, writes `model_export.json` (24 features)

### Dashboard (`index.html`, ~28.8 KB)
- [x] Inline `<style>` with frozen design tokens (Inter + IBM Plex Mono via Google Fonts CDN)
- [x] Inline `window.__MODEL__` payload + inline inference JS (zero external local refs)
- [x] Scaling -> calibrated probability -> risk gauge -> threshold verdict
- [x] Per-feature coefficient contribution bars
- [x] Counterfactual what-if controls
- [x] Opens via `file://`, no fetch

---

## 4. Verification Results

| Check | Result |
|---|---|
| Notebook code cells execute end-to-end (plain Python, Agg) | OK |
| Payload written | 24 features, threshold 0.355 |
| Test ROC-AUC / PR-AUC | 0.958 / 0.941 |
| Sensitivity / specificity (floor 0.90) | 0.964 / 0.727 |
| Browser-path sanity (high-risk patient) | p = 0.996349 -> HIGH |
| Browser-path low-risk patient | p = 0.016651 -> LOW |
| `index.html` external refs | Google Fonts only |
| File inventory | essentials only |

---

## 5. Acceptance Gates

1. **File Count Reduction** — PASS (8 files incl. dataset + docs).
2. **Dashboard Portability** — PASS (self-contained, `file://`, no local refs).
3. **Aesthetic Parity** — PASS (frozen tokens inlined).
4. **Notebook Integrity** — PASS (executes clean cell 1 -> N).
5. **Clinical Fidelity** — PASS (preprocessing, Platt calibration, sens >= 0.90 threshold, attribution all intact).
