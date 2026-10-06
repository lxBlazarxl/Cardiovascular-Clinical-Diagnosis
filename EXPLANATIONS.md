# EXPLANATIONS.md — How this project works, start to finish

This document explains the whole project: what the data means, what every column contributes,
what the notebook does at each step, how to run and check it, and what every control on the
dashboard does to the result. It also has an honest section on what the model **cannot** do.

Read `README.md` first for the short version. This is the long version.

---

## 1. The problem, in plain words

A doctor can measure many things about a patient cheaply: age, blood pressure, cholesterol, the
fastest heart rate the patient could reach on a treadmill, and so on. The one thing the doctor
really wants to know is expensive and invasive: **does this patient have a significant blockage
in the arteries that supply the heart?**

This project builds a cheap **first-pass risk estimate** so the doctor knows who should be sent
for the invasive test first.

It is **not** a diagnosis. It is a research and teaching prototype. See section 11.

### How the computer learns

We have 303 past patients. For each one we know all their measurements **and** whether they
really had a blockage. That second part is the answer key. We show the computer the measurements
and the answers, let it find the repeating pattern, then hand it a brand-new patient (no answer)
and let it make its best guess. This is **machine learning** — not magic, just careful arithmetic
on many examples.

### What "probability 0.85" actually means

It does **not** mean "this patient is 85% sick." It means: *among all patients who looked like
this one, about 85% had a blockage.* It is a statement about a group of similar people, not a
promise about one individual. This distinction matters more than anything else on the page.

---

## 2. The files

| File | What it is |
|---|---|
| `honors_mp.ipynb` | Deliverable A. Loads the data, explores it, trains the model, tunes the threshold, writes the model file, and self-checks. |
| `index.html` | Deliverable B. A single self-contained web page. Open it by double-clicking. No server, no install, no internet needed (except web fonts). |
| `model_export.json` | Written by the notebook. Every number the browser needs. Also inlined into `index.html`, so the page works even if this file is deleted. |
| `data/raw/cleveland.csv` | The input data. 303 lines, no header row, some values written as `?`. |
| `README.md` | Short orientation. |
| `OVERHAUL.md` | Record of the project restructure and its acceptance checks. |
| `requirements.txt` | The six Python libraries used. |

The whole point of the restructure was that there are only **two things a user needs**:
the notebook that builds the model, and the web page that uses it.

---

## 3. The data, column by column

The file is `data/raw/cleveland.csv`. It is a well-known public heart-disease dataset: 303
patients who visited a clinic in Cleveland, Ohio, in the late 1980s.

Two quirks we have to handle:

1. **There is no header row.** The file starts straight into data. So the notebook supplies the
   column names itself.
2. **Missing values are written as `?`.** A computer would read `?` as text, not as "unknown",
   and the arithmetic would break. The notebook tells pandas to treat `?` as a blank.

Here is the first real line of the file:

```
63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
```

### 3.1 The fourteen columns

| # | Column | Plain meaning | Values seen | Missing | Typical value |
|---|---|---|---|---|---|
| 1 | `age` | Age in years | 29 – 77 | 0 | 56 |
| 2 | `sex` | Biological sex | 1 = male, 0 = female | 0 | 1 (68% male) |
| 3 | `cp` | Chest pain type the patient reported | 1, 2, 3, 4 | 0 | 3 |
| 4 | `trestbps` | Resting blood pressure, mmHg (the higher number) | 94 – 200 | 0 | 130 |
| 5 | `chol` | Cholesterol, mg/dL | 126 – 564 | 0 | 241 |
| 6 | `fbs` | Fasting blood sugar above normal | 1 = yes, 0 = no | 0 | 0 (only 15% yes) |
| 7 | `restecg` | Resting ECG result | 0, 1, 2 | 0 | 1 |
| 8 | `thalach` | Highest heart rate reached during the treadmill test | 71 – 202 | 0 | 153 |
| 9 | `exang` | Did exercise cause chest pain? | 1 = yes, 0 = no | 0 | 0 |
| 10 | `oldpeak` | ST depression from the treadmill test — a heart-strain measure | 0 – 6.2 | 0 | 0.8 |
| 11 | `slope` | Shape of the ST segment during exercise | 1, 2, 3 | 0 | 2 |
| 12 | `ca` | Number of major vessels found blocked (fluoroscopy) | 0, 1, 2, 3 | **4** | 0 |
| 13 | `thal` | Thallium scan result | 3, 6, 7 | **2** | 3 |
| 14 | `target` | The answer key: severity of blockage, 0 – 4 | 0, 1, 2, 3, 4 | 0 | 0 |

### 3.2 The coded categories in full

**`cp` — chest pain type.** This is the patient's *description*, not a severity scale:

- `1` typical angina — chest pain that classically squeezes the chest during exertion
- `2` atypical angina — pain that is chest-related but not the classic pattern
- `3` non-anginal pain — chest pain that does not behave like heart pain
- `4` asymptomatic — no chest pain at all

**`restecg` — resting ECG:**

- `0` normal
- `1` ST-T wave abnormality (a mild, non-specific change)
- `2` left ventricular hypertrophy (the main heart muscle is thickened/enlarged)

**`slope` — shape of the ST segment during peak exercise:**

- `1` upsloping — usually reassuring
- `2` flat — intermediate
- `3` downsloping — most concerning, suggests blood flow trouble

**`thal` — thallium scan (a radioisotope scan of heart muscle blood supply):**

- `3` normal — all muscle gets blood
- `6` fixed defect — a scarred area that never recovers (past heart attack)
- `7` reversible defect — an area that stops getting blood under stress

### 3.3 The answer column

`target` is a **severity score from 0 to 4**, not a yes/no. We want yes/no, so the notebook does:

```python
raw["target"] = (raw["target"] > 0).astype(int)
```

- score `0` → `0` (no significant blockage)
- score `1, 2, 3, 4` → `1` (there is a blockage)

This gives **164 patients with no disease and 139 with disease** — a 45.9% positive rate. That is
a nicely balanced dataset. If 99% were healthy, a model that always answered "healthy" would
score 99% and still be useless, so a balanced split matters for the accuracy number to mean
anything.

### 3.4 The two columns with missing values

`ca` is missing for 4 patients and `thal` for 2. We fill them in during preprocessing (section
5.3) rather than dropping the patients, because 6 patients out of 303 is 2% — throwing away
whole rows for two blanks would cost us real information.
---

## 4. What each column actually contributes

Two different questions here, and they do not have the same answer.

### 4.1 Which columns carry signal on their own?

We fed each column *by itself* to a scoring function and measured how well it ranked sick patients
above healthy ones. AUC 0.5 means "no better than a coin flip." Higher is better; lower than 0.5
means the column is informative but points the opposite way.

| Column | Standalone AUC | Strength | Verdict |
|---|---|---|---|
| `thal` | 0.766 | strong | best single column |
| `cp` | 0.754 | strong | chest pain type is highly informative |
| `ca` | 0.750 | strong | vessels blocked — nearly a direct measurement |
| `thalach` | 0.255 | strong | 0.255 means low peak heart rate → disease; i.e. strength 0.745 |
| `oldpeak` | 0.735 | strong | exercise-induced heart strain |
| `exang` | 0.703 | good | exercise causes chest pain |
| `slope` | 0.689 | good | |
| `age` | 0.637 | moderate | |
| `sex` | 0.630 | moderate | |
| `restecg` | 0.585 | weak | |
| `trestbps` | 0.574 | weak | blood pressure alone says little |
| `chol` | 0.570 | weak | cholesterol alone says little |
| `fbs` | 0.509 | **almost none** | barely better than a coin flip |

Two things worth noticing:

- **Cholesterol and blood pressure are weak on their own.** That surprises people, but it is
  real: in this dataset they are noisy measurements, and a single cholesterol number is a poor
  predictor of blockage. They still help a little once the model weighs everything together.
- **`fbs` is nearly useless.** Its coefficient ends up slightly *negative*. We keep it because
  removing inputs is a modelling choice the user asked us not to make, and because it is a real
  clinical variable — but do not read anything into it.

### 4.2 What the trained model does with each column

After training, each feature gets a **coefficient** — a weight. All 13 inputs expand into 24
features, and here is every one with its learned weight.

Read the sign as "pushes risk **up**" (+) or "pushes risk **down**" (−).

| # | Feature | Weight | Direction |
|---|---|---|---|
| 0 | `age` | −0.061 | slightly **lowers** risk ← see section 11.1, this is a bug in the model, not real medicine |
| 1 | `trestbps` | +0.414 | higher blood pressure → more risk |
| 2 | `chol` | +0.211 | higher cholesterol → more risk |
| 3 | `thalach` | −0.241 | higher peak heart rate → less risk (protective) |
| 4 | `oldpeak` | +0.181 | more exercise-induced strain → more risk |
| 5 | `ca` | **+1.137** | blocked vessels → more risk, **by far the strongest** |
| 6 | `rpp` (engineered) | −0.101 | see 4.3 |
| 7 | `hr_reserve` (engineered) | −0.046 | see 4.3 |
| 8 | `cp_1` typical angina | −0.657 | |
| 9 | `cp_2` atypical angina | +0.162 | |
| 10 | `cp_3` non-anginal pain | −0.473 | |
| 11 | `cp_4` asymptomatic | **+0.967** | silent chest pain is the most dangerous presentation |
| 12 | `restecg_0` normal | −0.155 | |
| 13 | `restecg_1` ST-T abnormality | −0.071 | |
| 14 | `restecg_2` LV hypertrophy | +0.226 | |
| 15 | `slope_1` upsloping | −0.499 | |
| 16 | `slope_2` flat | +0.620 | |
| 17 | `slope_3` downsloping | −0.121 | |
| 18 | `thal_3` normal | −0.540 | |
| 19 | `thal_6` fixed defect | −0.252 | |
| 20 | `thal_7` reversible defect | **+0.791** | stress-only blood deficit is the bad one |
| 21 | `sex` | +0.726 | male → more risk in this dataset |
| 22 | `fbs` | −0.108 | essentially noise (see 4.1) |
| 23 | `exang` | +0.295 | exercise-induced chest pain → more risk |

Plus a starting offset, the **intercept** of −0.460, which is the log-odds of "no disease" for a
patient sitting at the average of every feature.

### 4.3 The two features we invent

Thirteen raw columns become 24 features. Ten of the extra eleven are just one-hot expansions of
the four category columns. The two genuinely *new* numbers are:

**`rpp` — rate-pressure product** = `trestbps × thalach ÷ 100`

Blood pressure times peak heart rate, divided by 100 for a readable number. It is a rough estimate
of how hard the heart is working during peak exertion. Typical value in our data: 198.

**`hr_reserve` — heart-rate reserve** = `thalach ÷ (220 − age)`

`220 − age` is a common rule of thumb for the maximum heart rate someone of that age should be
able to reach. Dividing the patient's actual peak by that estimate says **how close they got to
their own expected maximum** — roughly, how much cardiovascular reserve they have left.
Typical value: 0.94, meaning "they reached 94% of their predicted maximum."

Both formulas can divide by a zero or a near-zero (an age of 220, a peak rate of 0), which
produces infinity — a number that quietly poisons all downstream math. The code replaces infinity
back with "missing" so the fill-in step can handle it normally.

Both ended up with small coefficients, which is worth being honest about: the model did not find
them to be very useful. They are still there because they are standard clinical ideas and cost
nothing.

### 4.4 A note on `oldpeak` and `slope`

These two are **partly the same measurement** — both describe the ST segment during the treadmill
test — and they correlate at **r = 0.58**. Patients with a flat or downsloping ST segment have
higher `oldpeak` on average (mean `oldpeak` is 0.41 when upsloping, 1.43 when flat, 2.71 when
downsloping). So when the dashboard shows "ST slope" as a driver, some of that is really the same
underlying treadmill finding already counted through `oldpeak`. The model handles this fine
because both are regularised, but it means the two sliders are not fully independent.
---

## 5. The notebook, step by step

The notebook is `honors_mp.ipynb`: 37 cells, 8 steps. Every markdown explanation in it is written
in plain language for an eighth-grader. Here is what each step does and, just as importantly,
what it deliberately does **not** do.

### Step 1 — Load the data

Reads `data/raw/cleveland.csv`, supplies the column names, converts `?` to real blanks, and turns
the 0–4 `target` into a 0/1 answer.

- **Does:** attach names, mark missing values, binarise the target, confirm 303 rows.
- **Does not:** drop any rows, impute anything, or touch the target distribution yet.

### Step 2 — Look at the data (EDA)

"EDA" means *Exploratory Data Analysis*: spend time looking before you build. It is like checking
your fridge before cooking, because building a model on garbage and never noticing is the most
common and most avoidable failure.

Six looks:

- **2a** `.info()` — row counts and types per column; spots the two columns with blanks.
- **2b** missing-value count per column, worst first.
- **2c** `.describe()` — count, mean, standard deviation, min, quartiles, max per column.
- **2d** class balance — the 164/139 split, as a bar chart and a pie chart.
- **2e** histograms per numeric column, coloured by class. If the two colours sit in different
  places, the column separates the groups.
- **2f** correlation heatmap — which columns move together.
- **2g** box plots, healthy vs sick side by side, so you can see which columns separate cleanly.

- **Does:** describe and visualise. Nothing is fitted or changed.
- **Does not:** make any decision. These cells are for a human to read.

### Step 3 — Prepare the data (preprocessing)

Raw data cannot go straight into a model for three reasons, all fixed here.

#### 5.1 Two new features

`rpp` and `hr_reserve`, as defined in section 4.3.

#### 5.2 The three kinds of column, three recipes

**Numeric** — real measured numbers: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`, plus
the two engineered ones.

- *Fill blanks with the **median*** (the middle value). The median is used instead of the average
  because one extreme patient would drag an average around, but cannot move a median much.
- *Then **standardise***: subtract the average, divide by the spread. After this every numeric
  column sits near 0 with a similar range, so no column can bully the others. Cholesterol (~247)
  and `oldpeak` (~1.0) become comparable numbers.

**Categorical** — labels, not quantities: `cp`, `restecg`, `slope`, `thal`.

- *Fill blanks with the **most common value***.
- *Then **one-hot encode***: split one column into several yes/no columns. `cp` becomes
  `cp_1`, `cp_2`, `cp_3`, `cp_4` — exactly one is 1 for each patient.

  This matters because it stops the model assuming type 4 is "four times as bad as" type 1.
  Each chest-pain type is its own separate thing.

**Binary** — already yes/no: `sex`, `fbs`, `exang`.

- *Fill blanks with the most common value, then standardise* like the numeric ones.

So 13 raw inputs become **24 features**: 8 numeric (6 raw + 2 engineered), 13 one-hot columns
(4 + 3 + 3 + 3), and 3 binary.

#### 5.3 The golden rule: never look at the test data

Every median, mode, average and spread is learned **from the 242 training patients only**, then
applied unchanged to the 61 test patients. Learning a fill value from the test patients is a form
of cheating — it makes the final score look better than it really is.

- **Does:** fill, scale, one-hot, using training statistics only.
- **Does not:** fit anything on the test split. Ever.

### Step 4 — Train the model

#### 5.4 The one model: logistic regression

Despite the name it is simple:

1. Multiply each of the 24 features by its weight and add them up, plus the intercept. This total
   is the **score** (log-odds).
2. Positive score leans toward "sick", negative toward "healthy".
3. Squeeze the score through an **S-curve** (the sigmoid) to get a probability between 0 and 1.

Training is just searching for the 24 weights that make the best guesses on the training data.
Two settings:

- `C=1.0` — how strictly we penalise large weights. Large weights mean memorising the training
  patients instead of learning a pattern that generalises.
- `max_iter=2000` — maximum number of attempts, set high so it definitely finishes.

#### 5.5 Splitting off the final exam

20% of patients (61) are held back as the **test set** and never trained on. The split keeps the
same sick/healthy ratio in both halves (a *stratified* split) and fixes a seed (`RANDOM_STATE=42`)
so the split is identical on every run. That leaves **242 train / 61 test**.

#### 5.6 Calibration: making the probabilities honest

A raw logistic score is not automatically a trustworthy probability. If the model says "70% risk"
we would like roughly 70 of 100 such patients to really be sick. Often they do not — the model is
overconfident.

**Platt scaling** fixes this with one extra small S-curve that remaps the raw score into a
better-behaved probability. Our curve is `p = sigmoid(0.890 × score − 0.042)`.

**The subtle part** — we fit that curve using **out-of-fold** scores, not the final model's own
scores. Concretely:

1. Split the training patients into 5 folds.
2. Train on 4 folds, score the held-out fold. Repeat so every fold is scored by a model that
   never saw it.
3. Fit the calibration curve on those 5 honest out-of-fold score sets.

If we instead calibrated on the final model's own training scores, the model would look more
confident than it deserves, and we would be calibrating against predictions that were already
too good.

- **Does:** split, train, calibrate out-of-fold, refit on all training data.
- **Does not:** touch the test patients. Not for training, not for calibration.

### Step 5 — Pick the cutoff (threshold tuning)

The model produces a probability like 0.62, but the doctor needs a simple answer: high risk or
not. The **threshold** is the dividing line.

**Why not the usual 0.5?** Because the two kinds of mistake are not equally bad:

- a **false negative** (sick patient called low risk) means someone misses the test they needed;
- a **false positive** (healthy patient called high risk) means extra worry and maybe an
  unnecessary test.

A missed disease is worse, so we deliberately accept more false alarms.

**The exact rule, in order:**

1. Try every threshold from 0.050 to 1.000 in steps of 0.001.
2. Discard any threshold where **training sensitivity is below 0.90** — we must catch at least
   9 in 10 sick patients.
3. Among the survivors, keep the one with the **highest specificity** (fewest false alarms while
   still keeping the 90% promise). Ties go to the **lowest** threshold.

Chosen value: **0.355**, giving 90.1% sensitivity and 80.9% specificity on the training set.

- **Does:** search thresholds on the **training** split only.
- **Does not:** look at the test split while choosing. See section 11.2 — an earlier version of
  this project did exactly that, and it made the numbers look better than they deserved.

### Step 6 — Check the results

The first time the model is graded on the 61 patients it has never seen.

Four pictures and two tables:

- **ROC curve** — sensitivity against false-alarm rate, for every possible threshold. AUC 0.958.
  The dashed diagonal is a coin flip.
- **Precision-Recall curve** — precision against recall. AUC 0.941.
- **Calibration curve** — predicted probability against how often it actually happened. Points
  near the diagonal mean the probabilities can be trusted.
- **Confusion matrix** — the four outcomes.
- `classification_report` — precision, recall, F1 per class, plus accuracy.

**Final held-out results (n = 61):**

| Metric | Value | Plain meaning |
|---|---|---|
| ROC-AUC | **0.958** | ranks sick patients above healthy ones very well |
| PR-AUC | **0.941** | same, focused on the sick class |
| Sensitivity | **0.964** | caught 27 of 28 sick patients |
| Specificity | **0.727** | correctly cleared 24 of 33 healthy patients |
| Brier score | **0.086** | average squared error of the probabilities; lower is better |
| Accuracy | 0.836 | 51 of 61 correct |
| Threshold | 0.355 | |

**The confusion matrix:**

|  | Predicted LOW | Predicted HIGH |
|---|---|---|
| **Actually healthy** (33) | 24 correct (TN) | 9 false alarms (FP) |
| **Actually sick** (28) | 1 missed (FN) | 27 caught (TP) |

Per class: healthy class precision 0.960 / recall 0.727; sick class precision 0.750 / recall
0.964. 59% of the test set is flagged HIGH while the true rate is 46% — the threshold is
deliberately liberal, trading false alarms for missed diagnoses.

- **Does:** evaluate and report.
- **Does not:** retune anything. If you do not like a number here, the fix is a modelling
  decision made in step 5, not an adjustment to the report.

### Step 7 — Save the model

Writes `model_export.json`, because the notebook runs in Python and the dashboard runs in
JavaScript with no Python at all. The file carries every number the browser needs:

| Key | What it holds |
|---|---|
| `feature_names` | the 24 features, in order |
| `coef`, `intercept` | the 24 learned weights and the offset |
| `calibrator_coef`, `calibrator_intercept` | the Platt curve |
| `threshold` | 0.355 |
| `num_mean`, `num_scale`, `num_median` | how each numeric column was filled and rescaled |
| `bin_mean`, `bin_scale`, `bin_median` | same for the three binary columns |
| `cat_cols`, `cat_categories` | the category columns and their one-hot labels |
| `metrics` | the step-6 scores shown on the dashboard |

**One trap worth naming.** A column that never changes has a spread of zero, and dividing by zero
blows up. scikit-learn quietly substitutes `1.0` in that case, so the export must too or the
browser would disagree with Python. (In this dataset `fbs` is nearly constant but not perfectly
so, so the substituted value is small rather than exactly 1.0.)

- **Does:** write a plain-text file of numbers.
- **Does not:** embed code. There is no `eval`, no pickles, nothing executable.

### Step 8 — Sanity check: does Python agree with the browser?

The last cell proves the two paths give identical answers, for the same patient:

1. **Python path** — push the patient through the real fitted pipeline. Ground truth.
2. **Browser path** — ignore every fitted object and rebuild the answer *using only the numbers
   in the JSON file*. This mirrors the JavaScript in `index.html` line for line.

The printed difference is **`0.00e+00`** — not small, exactly zero. If the dashboard ever drifts
from the notebook, this cell stops matching and you know to re-export.
---

## 6. How to run it

### 6.1 The dashboard (no install, 5 seconds)

Open `index.html` by double-clicking it, or:

```bash
xdg-open index.html      # Linux
open index.html          # macOS
```

That is the whole procedure. No server, no `pip install`, no internet. The page works from a
`file://` URL.

It will work **even if you delete `model_export.json`**, because the model numbers are inlined
into the HTML. You only need to delete that file if you want to be sure there is exactly one
source of truth; then run the notebook to regenerate it.

The only outside reference is the Google Fonts link for the Inter and IBM Plex Mono typefaces.
If you are fully offline the page still works — it just falls back to system fonts.

### 6.2 The notebook (needs Python)

```bash
uv venv .venv
uv pip install -r requirements.txt
```

`requirements.txt` installs the six *science* libraries only. It deliberately does **not** install
Jupyter, because Jupyter is a tool for running the notebook, not a dependency of the project. So
after a fresh clone you must add the kernel yourself:

```bash
uv pip install ipykernel jupyterlab
.venv/bin/python -m ipykernel install --user --name honors_mp --display-name "Python (honors_mp)"
```

The second command registers `.venv/bin/python` as a selectable kernel called **Python
(honors_mp)**. It writes the interpreter's path as an absolute path, so the kernel cannot
accidentally bind to a different Python on your system. The notebook's own metadata already
requests this kernel, so it is selected automatically when you open the file.

Then open it:

```bash
.venv/bin/python -m jupyter lab honors_mp.ipynb
```

...or open the file in VS Code (with the Jupyter extension) or any other Jupyter frontend and pick
**Python (honors_mp)** from the kernel menu.

> **Run all cells in order.** Step 4 needs step 3's fitted pipeline, and step 7 needs step 5's
> threshold. *Kernel → Restart Kernel and Run All* is the safe option.

### 6.3 Troubleshooting: "I can't run the notebook with my .venv kernel"

The exact symptom is a kernel menu that is empty, or an error like
`No such kernel named python3`, or a dialog offering no Python at all. The cause is almost always
that **`ipykernel` is not installed in the venv**, so there is nothing for the frontend to launch.

Diagnose it in one line:

```bash
.venv/bin/python -c "import ipykernel; print('ok')"
```

If that prints `ModuleNotFoundError: No module named 'ipykernel'`, that is your problem. Fix it
with the two commands in 6.2. You can confirm what the frontend can see with:

```bash
.venv/bin/python -m jupyter kernelspec list
```

You should see `honors_mp` in the output. **If you do not see it**, the kernel was never
registered, or it was registered for a different Python than the one you are running — re-run the
`ipykernel install` command from inside the project directory.

One more trap: a kernelspec named `python3` that resolves to a bare `python` (rather than an
absolute path) will bind to whatever Python happens to be first on your `PATH`, which is often
*not* your venv. That is why this project registers the explicit name `honors_mp` instead.

### 6.3 Running the notebook without Jupyter

Jupyter is not required. Save this as `run_notebook.py` in the project folder:

```python
import contextlib, io, json
import matplotlib
matplotlib.use("Agg")            # no display needed on a server

nb = json.load(open("honors_mp.ipynb"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]

ns = {"__name__": "__main__"}
for i, cell in enumerate(code):
    src = "".join(cell["source"])
    print(f"--- cell {i + 1}/{len(code)} ---")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(compile(src, f"<cell {i + 1}>", "exec"), ns)
    print(buf.getvalue().rstrip() or "(no output)")

print("\nmetrics:", ns["metrics"])
assert ns["metrics"]["sensitivity"] >= 0.90, "sensitivity floor violated"
print("sensitivity floor >= 0.90: PASS")
```

Run it with:

```bash
.venv/bin/python run_notebook.py
```

This is exactly how the notebook was verified. It rewrites `model_export.json`, so run it from
the project root.

---

## 7. How to check that everything is correct

Five checks, in increasing order of thoroughness. The first three take seconds.

### 7.1 Does the notebook run at all?

Run `run_notebook.py` (section 6.3). It should end with:

```
metrics: {'model': 'logistic_regression', 'threshold': 0.355, 'n_test': 61,
          'roc_auc': 0.9577922077922079, 'pr_auc': 0.9407883021856493,
          'sensitivity': 0.9642857142857143, 'specificity': 0.7272727272727273,
          'brier_calibrated': 0.08620931037001911}
sensitivity floor >= 0.90: PASS
```

Those numbers are deterministic — the seed is fixed, so they must match exactly. If they differ,
something changed that should not have.

### 7.2 Does the dashboard match the notebook?

The `index.html` payload must be byte-identical to `model_export.json`. Save this as
`check_sync.py`:

```python
import json, re

nb = json.load(open("model_export.json"))
html = open("index.html").read()
m = re.search(r"window\.__MODEL__\s*=\s*(\{.*?\});", html, re.S)
page = json.loads(m.group(1))

bad = [k for k in set(nb) | set(page) if nb.get(k) != page.get(k)]
print("keys:", len(nb), "| mismatches:", bad or "none")
print("EXACT MATCH" if not bad else "OUT OF SYNC -> re-run the notebook and re-inline")
```

If it says out of sync, run the notebook (it rewrites `model_export.json`) and then re-inline it
into the page with:

```python
import json, re
payload = json.load(open("model_export.json"))
compact = json.dumps(payload, separators=(",", ":"))
html = open("index.html").read()
html = re.sub(r"window\.__MODEL__\s*=\s*\{.*?\};",
              "window.__MODEL__ = " + compact + ";", html, count=1, flags=re.S)
open("index.html", "w").write(html)
```

### 7.3 Does the browser's JavaScript agree with scikit-learn?

The strongest check. It extracts the real `vector()` and `predict()` functions out of the shipped
HTML and runs them in Node against the fitted pipeline. This needs Node.js (`node --version`).

On the six standard test patients the two paths agree to **within 4.5e-17** — that is machine
epsilon, the last bit of a double-precision float, caused only by the two languages summing
floating-point numbers in a slightly different order. Four of the six cases match at exactly
`0.00e+00`; the other two differ in the 17th decimal place. Every verdict (HIGH/LOW) is
identical.

If you want the exact-zero check, use the in-notebook one instead: cell 8 compares the real
pipeline against the JSON-only reconstruction using numpy on both sides, and prints `0.00e+00`.

### 7.4 Spot-check the dashboard by hand

1. Open `index.html`.
2. The page should load showing the **default patient**: 58-year-old male, asymptomatic chest
   pain, blood pressure 158, cholesterol 285, peak heart rate 118, exercise angina on, 2 blocked
   vessels, thallium 7.
3. It should read **HIGH RISK, 99.6%**, with the referral threshold marked at 35.5% on the bar.
4. Click **Low-risk example**. It should read **LOW RISK, 1.7%** with the patient reset to a
   41-year-old female, peak heart rate 176, no blocked vessels, normal thallium.
5. Drag **Blocked vessels** from 0 up to 3 on the low-risk patient. Watch the number climb from
   1.7% to about 34.7% — the single biggest lever in the whole model.

### 7.5 Check the file inventory

The project should contain only these (plus `.git`, `.venv`, `.gitignore`):

```bash
ls -A | grep -v '^\.venv$\|^\.git'
```

Expect: `EXPLANATIONS.md`, `OVERHAUL.md`, `README.md`, `data`, `honors_mp.ipynb`, `index.html`,
`model_export.json`, `requirements.txt`.

---

## 8. The dashboard, control by control

The page is two columns. Left: patient inputs. Right: the verdict, a risk bar, feature
attributions, and a what-if panel. Everything recomputes instantly as you move a control — there
is no submit button, because there is nothing to send anywhere.

### 8.1 How a control becomes a number

Every control feeds the same five steps, in JavaScript, in your browser:

1. **Compute the two engineered features** — `rpp = trestbps × thalach ÷ 100` and
   `hr_reserve = thalach ÷ (220 − age)`.
2. **Standardise** the numeric values using the saved `num_mean` and `num_scale`, and the binary
   values using `bin_mean` and `bin_scale`. This is the exact arithmetic the notebook did.
3. **One-hot expand** the four category controls into their 13 yes/no columns.
4. **Take the weighted sum** — dot product of the 24 standardised values with the 24 `coef`
   values, plus the intercept.
5. **Calibrate and compare** — push the score through the Platt curve to get the probability, then
   compare with the 0.355 threshold.

The per-feature contributions shown in the "Feature attributions" tab are each
`coefficient × standardised value`. They are the actual additive pieces of the score, so they
genuinely add up to the total.

### 8.2 The three verdict bands

| Band | Condition | Meaning shown |
|---|---|---|
| **HIGH RISK** | probability ≥ 0.355 | "This profile meets the referral threshold." |
| **MODERATE RISK** | 0.25 ≤ probability < 0.355 | "Below the referral threshold but worth monitoring." |
| **LOW RISK** | probability < 0.25 | "Well below the referral threshold." |

Note that only **0.355** is the real decision boundary. The 0.25 cut and the MODERATE band are
**display only** — a visual cushion so a patient just under the line does not look identical to a
very low-risk patient. Nothing in the model or the exported file depends on 0.25.

### 8.3 Every control: range, direction, and how much it moves the number

Measured by moving one control at a time and leaving the other twelve at the low-risk example's
values. "Span" is how much the probability moves across the control's full range.

| Control | Range | Effect on risk | Span | Notes |
|---|---|---|---|---|
| **Age** | 20 – 100 | **slightly *decreases*** | 0.010 | ⚠️ Backwards. See 11.1. |
| **Biological sex** | Male / Female | Male **+** | 0.047 | |
| **Chest pain type** | 4 options | Asymptomatic **+** most | 0.025 | Typical angina is the *lowest* |
| **Resting blood pressure** | 80 – 220 mmHg | **increases** | 0.078 | Strong, steady |
| **Cholesterol** | 100 – 600 mg/dL | **increases** | 0.060 | Strong, steady |
| **Fasting blood sugar** | Yes / No | negligible | 0.004 | Near-useless column |
| **Resting ECG** | 3 options | LV hypertrophy **+** | 0.007 | Small |
| **Max heart rate** | 60 – 220 bpm | **strongly decreases** | 0.070 | Protective. A fit patient who can reach a high max HR has more reserve. |
| **Exercise-induced angina** | Yes / No | Yes **+** | 0.012 | |
| **ST depression** | 0 – 7.0 | **increases** | 0.028 | More exercise-induced strain |
| **ST slope** | 3 options | Flat **+** most | 0.027 | Not fully independent of ST depression (4.4) |
| **Blocked vessels** | 0 – 3 | **increases, steeply** | **0.331** | ⚠️ **By far the strongest control.** |
| **Thallium scan** | 3 options | Reversible defect **+** most | 0.036 | |

**The single most important thing to understand:** `ca` (blocked vessels) is the strongest
*individual* control. On the low-risk example, moving it from 0 to 3 takes the probability from
1.7% to 34.7% — a swing of 0.331, more than four times the next-largest control (blood pressure,
0.078). This makes clinical sense: `ca` is close to a direct measurement of the thing we are
predicting, whereas blood pressure and cholesterol are indirect risk factors.

But it is **not** the only thing that matters, and the model is not simply reading `ca`. If you
push every *other* control to its own worst setting at once, the probability goes from 1.7% to
**99.9%** — a far larger jump than `ca` alone. Risk accumulates: a model this bad only shows you
the danger when several things go wrong together, which is also how it works in real medicine.

### 8.4 The what-if panel

Two extra sliders — target blood pressure (90–180) and target cholesterol (120–320) — let you ask
"what if this patient changed these two things?" The panel shows a **current** bar and a
**modelled** bar, and states the change in plain words.

**How to read it, and the trap:** risk falls when you set the targets **below** the patient's
current values, and rises when you set them above. The sliders default to 140/200. For the
low-risk example (118/168) those defaults are *higher* than the current values, so the panel will
tell you those targets would *raise* modelled risk — which is correct, just not what people
expect on first look. **Always lower the sliders below the patient's current values to see a
benefit.**

The panel is honest about its own limits: it says "modelled", not "predicted outcome". Changing
blood pressure in a slider is not the same as a patient actually changing their blood pressure.

### 8.5 The two sample buttons

- **High-risk example** → the default patient above (99.6%, HIGH).
- **Low-risk example** → 41-year-old female, 118/168, peak HR 176, no angina, 0 vessels, normal
  thallium (1.7%, LOW).

Use **Low-risk example** when exploring controls, because the high-risk patient is so far above
the threshold that almost nothing moves the verdict.

---

## 9. What the numbers in the dashboard mean

- **Model probability** — the calibrated chance this patient has a blockage, 0–100%.
- **Referral threshold** — 35.5%, our tuned cutoff. The tick mark on the risk bar.
- **Feature attributions** — the top 10 features by absolute contribution to the score, green bars
  pushing risk up, red bars pushing it down, with the signed number.
- **Increasing / Decreasing risk** — a one-line plain-English summary of the top three in each
  direction.
- The disclaimer at the bottom is not decoration. See section 11.
---

## 10. Design decisions worth knowing

**Why one model and not several?** The brief was deliberately simple. A single logistic
regression can be read on one page: 24 weights, an intercept, and an S-curve. Anything stronger
(gradient boosting, an ensemble, feature selection) would improve the score by a fraction of a
point while making the model impossible to explain to a clinician. The ROC-AUC of 0.958 is well
into the "genuinely useful" range, so the extra complexity would not be earning its keep.

**Why the threshold is 0.355 and not 0.5?** Section 5.5. Because a missed diagnosis costs more
than a false alarm, and 0.5 is a software default with no medical meaning.

**Why is the specificity only 0.727?** Because we deliberately traded it away to keep sensitivity
at 0.964. That is the intended behaviour of the design, not a defect. If you would rather have
fewer false alarms, raise the threshold — but sensitivity will drop below 0.90 and the clinical
promise in the brief is broken.

**Why keep `fbs` if it does nothing?** Because removing inputs is a modelling decision that was
explicitly out of scope. It is a real clinical variable; it just happens to be uninformative in
this dataset.

**Why two engineered features if the model barely uses them?** `rpp` and `hr_reserve` are
standard clinical ideas, they cost nothing, and a model that assigns them small weights has
correctly declined to over-rely on them. Keeping them documents the reasoning.

---

## 11. What this model cannot do

This is the most important section in the document. Please do not skip it.

### 11.1 ⚠️ The age coefficient is backwards

In the trained model, **older age slightly *lowers* predicted risk** (weight −0.061). That is
clinically wrong, and it is a genuine defect.

The cause is visible in the data. Older patients in this dataset reach a **lower** peak heart rate
(patients over 70 average 136 bpm; patients under 40 average 169), and peak heart rate is one of
the strongest protective factors in the model. The model has partly learned "lower peak heart
rate" and, in doing so, absorbed some of the real effect of age into the wrong bucket. The
`hr_reserve` feature, which divides peak heart rate by `220 − age`, compounds the inversion.

A small-sample effect makes it worse at the top of the range: the 70–78 age band contains only
**6 patients**, and just one of them had disease. The model sees "very old" as "very low risk"
based on essentially no evidence.

Standalone, age is genuinely informative (AUC 0.637, and the disease rate rises from 30% in the
40–50 band to 59% in the 60–70 band). So the signal is real; the model has just used it badly.

**What to do about it:** on the dashboard, **treat the age slider as unreliable**. It moves the
number by only 0.010 across its entire 20–100 range, and in the wrong direction. If age matters
to your judgement, weigh it yourself rather than reading it off this model.

### 11.2 ⚠️ The threshold was once chosen using the test data

Worth recording, because it is a real mistake that was found and fixed.

The project's earlier version reported **threshold 0.416, sensitivity 0.964, specificity 0.818**.
Those numbers were produced by running the threshold search on the **test** patients. It is not a
subtle bias: choosing a cutoff on the same patients you then report results for always flatters
the result. That inflated specificity from **0.727 to 0.818** — a jump of nearly nine percentage
points, entirely artificial.

The notebook now searches on the training split only, giving the honest **0.355 / 0.964 /
0.727**. The reported sensitivity is unchanged at 0.964, so the clinical promise still holds; you
simply give up some specificity, which is the correct price of not cheating.

The dashboard, `model_export.json`, `README.md` and `OVERHAUL.md` were all updated together. If
you find the old numbers quoted anywhere, they are stale.

### 11.3 Calibration barely helps on this dataset — and here it very slightly hurts

Platt scaling is retained, but be clear-eyed about it:

| | Brier score (lower better) | ROC-AUC |
|---|---|---|
| Uncalibrated | **0.0855** | 0.9578 |
| Calibrated | 0.0862 | 0.9578 |

The calibrated Brier score is very slightly **worse**. And ROC-AUC is *identical*, which is
expected: calibration is a monotone remapping, so it cannot change the ranking of patients — only
where the numbers sit and whether 0.8 honestly means 80%.

So on 61 test patients, calibration bought nothing measurable. It is kept because the machinery
is already there, it is the standard approach, and on a larger set it would likely help. But you
should not expect it to be doing heavy lifting. The `brier_calibrated` number in the payload is
the honest one to quote.

### 11.4 Small sample, old data, one hospital

- **303 patients, 61 in the test set.** When the sick group in the test set is 28 people, one
  patient changing class moves sensitivity by 3.6 percentage points. Every number here has wide
  error bars and none of the third decimal places mean anything.
- **Data from a single clinic in Cleveland, late 1980s.** Treatments, thresholds and population
  have all moved on since. Do not assume it transfers to your patients.
- **No external validation.** The model was never tested on any other dataset, hospital or
  population. It has only ever been checked on 61 patients from the same 303 it grew from.
- **The target is a 1980s angiographic criterion.** The conventional description of this dataset
  defines "disease" as narrowing of at least 50% in a major vessel, measured by angiography.
  Modern practice has a much lower threshold for investigation, so a "negative" here is not the
  same as "no heart disease" today.
- **`thal` and `ca` are results of scans most patients would not have.** Two of the strongest
  predictors in the model come from fluoroscopy and thallium imaging. In practice these are
  exactly the tests you are trying to avoid. The model's real-world value therefore depends on
  how well it performs *without* them — which this model was not separately evaluated for.

### 11.5 The things it structurally cannot do

- **It cannot diagnose anyone.** A probability is a group statement (section 1).
- **It cannot tell you *why* this patient is high risk in a causal sense.** On the low-risk
  example with 3 blocked vessels, the dashboard credits "blocked vessels" with `+3.089` toward
  the risk score. That is an association measured across 303 records — not a claim that
  unblocking those vessels would prevent a heart attack.
- **It cannot extrapolate beyond the slider ranges.** The dashboard allows age 20–100, but the
  data only covers 29–77. Age 20 and age 100 are guesses, and age 100 is exactly where 11.1
  bites hardest.
- **It cannot handle missing data.** The notebook imputes medians; the dashboard cannot express
  "unknown" at all. A patient whose `ca` was never measured looks identical to a patient with
  zero blocked vessels — and since `ca` is the strongest predictor, that is the single most
  misleading thing the dashboard can do. If a value is genuinely unknown, do not enter 0.
- **It does not model interactions.** Effects are assumed additive. "High cholesterol *and* high
  blood pressure" is scored as the plain sum of the two, with no extra penalty or synergy.
- **It will never refuse a patient.** It always produces a number between 0% and 100%. There is
  no "this patient is outside what I can help with" state, and for inputs far outside the
  training range it will confidently return a number that means nothing.
- **It is not approved for anything clinical.** The page says so, and it is not boilerplate.

### 11.6 Reproducibility caveats

- Results are deterministic only because `RANDOM_STATE=42` is fixed. Change it and every number
  in this document changes slightly.
- `requirements.txt` pins exact versions. **Check them against your environment before trusting a
  rerun.** The published numbers were produced with `numpy 2.5.3`, `pandas 3.0.6`,
  `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `seaborn 0.13.2`, `joblib 1.6.0`; if the pins in that
  file ever drift from those, regenerate `model_export.json` and re-run checks 7.1–7.3 before
  trusting the metrics.
- scikit-learn in particular changed `OneHotEncoder`'s `sparse` argument to `sparse_output`, and
  moving between major versions may alter one-hot column naming (`cp_1.0` vs `cp_1`). If the
  export and the dashboard stop matching after a dependency upgrade, run check 7.2.
- The notebook is stored with cleared outputs. Running it will populate them, which makes the
  file larger but shows no extra information.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Logistic regression** | A model that multiplies inputs by weights, adds them up, and squashes the total through an S-curve into a 0–1 probability. |
| **Coefficient / weight** | How much one feature pushes the risk score up or down. |
| **Intercept** | The starting score before any feature is counted. |
| **Score / log-odds** | The raw number before the S-curve. |
| **Sigmoid / S-curve** | `1 / (1 + e^-x)`, which turns any number into a probability between 0 and 1. |
| **Probability** | A number between 0 and 1 meaning "how often patients like this one had the condition." |
| **Threshold** | The cutoff where the probability becomes a HIGH/LOW decision. Here: 0.355. |
| **Sensitivity** | Of the truly sick, how many we caught. Higher means fewer missed diagnoses. |
| **Specificity** | Of the truly healthy, how many we correctly cleared. Higher means fewer false alarms. |
| **Calibration / Platt scaling** | Adjusting raw scores so a stated probability actually matches the observed frequency. |
| **Standardisation** | Subtracting the average and dividing by the spread, so no feature dominates by scale alone. |
| **One-hot encoding** | Turning a label column into several yes/no columns, so the model does not assume an order. |
| **Imputation** | Filling a missing value with a sensible stand-in. Here: the median, or the most common value. |
| **Median** | The middle value when sorted. Used instead of the mean because it is not dragged around by outliers. |
| **Pipeline** | A chain of steps run in order, so the same processing always happens the same way. |
| **Data leakage** | Letting information from the test set influence how the model is built. It makes results look better than they are. See 11.2. |
| **Out-of-fold prediction** | Predicting each training patient with a model that did not train on them, so the scores are honest. |
| **Cross-validation** | Repeatedly splitting data to train and test on different parts, to check stability. |
| **Test set** | Data held back and never trained on, used once at the end to measure real performance. |
| **ROC-AUC** | A single number for how well the model ranks sick above healthy, across all thresholds. 0.5 is a coin flip. |
| **Precision** | When the model says sick, how often it is right. |
| **Recall** | How many of the sick patients the model found. Same idea as sensitivity. |
| **F1** | A single blend of precision and recall. |
| **Brier score** | Average squared gap between predicted probability and what happened. Lower is better; it measures honesty, not just correct yes/no answers. |
| **False positive** | Told "high risk" but actually healthy. A false alarm. |
| **False negative** | Told "low risk" but actually sick. The dangerous miss. |
| **EDA** | Exploratory Data Analysis — looking at the data before modelling. |
| **JSON** | A plain-text format for storing numbers. `index.html` embeds one. |
