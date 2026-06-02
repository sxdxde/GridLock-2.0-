# Traffic Management & Travel Demand Forecast
### Flipkart Grid 6.0 — Competition Solutions

**Final Leaderboard Score: 91.50 / 100**  
**Metric:** `max(0, 100 × R²)`  
**Model:** LightGBM (3-seed bagged) | **Submission:** `submission_v9_bag.csv`

---

## Table of Contents
1. [Problem Understanding](#1-problem-understanding)
2. [Data Analysis & Key Observations](#2-data-analysis--key-observations)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture](#4-model-architecture)
5. [Validation Strategy](#5-validation-strategy)
6. [Experiment History & What Failed](#6-experiment-history--what-failed)
7. [Final Pipeline](#7-final-pipeline)
8. [Tools & Environment](#8-tools--environment)
9. [How to Reproduce](#9-how-to-reproduce)

---

## 1. Problem Understanding

The task is to predict normalised traffic **demand** (a continuous value in [0, 1]) for ~41,800 road segments, given their geohash location, road properties, and the time of day.

**Target:** `demand` — bounded in [0, 1], right-skewed (median ≈ 0.048, mean ≈ 0.094).  
**Metric:** `max(0, 100 × R²)` — penalises variance not explained by the model, rewards smooth fits.

### Dataset structure

| Split | Days | Time window | Rows |
|-------|------|------------|------|
| Train | Day 48 (full day) + Day 49 (00:00–02:00) | tmin 0–1425 | 77,299 |
| Test  | Day 49 | 02:15–13:45 (tmin 135–825) | 41,778 |

The fundamental insight is that the problem reduces to **interpolating a known spatiotemporal demand surface** forward in time. Day 48 provides a dense 24-hour snapshot of every geohash; the test set asks for predictions on a future 12-hour window of the same locations.

---

## 2. Data Analysis & Key Observations

### Temporal structure
- Training covers day 48 fully (96 time slots × ~723 geohashes) plus the first 9 slots of day 49.
- Test is entirely within day 49, from 02:15 to 13:45 — a **future window**.
- 99.94% of test geohashes appear in the training data.

### Critical OOD discovery: `abs_time`
An absolute time feature (`day × 1440 + tmin`) looks useful — it captures the day-level trend. But:

```
Training abs_time range: 0 – 69,240  (day 48 start → day 49, 02:00)
Test abs_time range:    69,255 – 69,945  (day 49, 02:15 → 13:45)
```

Every single test row has `abs_time` **beyond** the training maximum. Any tree model clips these to the same terminal leaf — the feature becomes a constant for all test rows. Using it caused an **87.1% LB score** despite a strong OOF.

### Geohash properties
- 1,249 unique geohashes in training; 1,248 appear in test.
- Geohash is a base-32 encoded geographic cell (precision 6 ≈ 1.2 km × 0.6 km).
- Decoding to (lat, lon) gives a continuous spatial representation: lat ∈ [−5.49, −5.24], lon ∈ [104.77, 105.01] — a roughly 30 × 30 km area.

### Missing values

| Feature | Train missing | Test missing |
|---------|--------------|-------------|
| RoadType | 600 (0.78%) | 324 (0.78%) |
| Temperature | 2,495 (3.23%) | 1,349 (3.23%) |
| Weather | 797 (1.03%) | 431 (1.03%) |

Missing rates are nearly identical between train and test — structurally missing, not informatively missing.

---

## 3. Feature Engineering

All features are **deterministic and leakage-free** — no target statistics are used anywhere in the final model.

### Spatial features
| Feature | Construction | Rationale |
|---------|-------------|-----------|
| `lat` | Base-32 geohash decode → latitude centre | Continuous spatial coordinate |
| `lon` | Base-32 geohash decode → longitude centre | Continuous spatial coordinate |
| `gh6` | Raw geohash string as native LGBM categorical | Lets the model learn each geohash's full demand profile without all-day-mean bias |

**Why `gh6` instead of target encoding:**  
Target encoding encodes the geohash's *all-day average demand*. Since test rows are only from 02:15–13:45, this all-day mean is a biased proxy for the daytime-only signal. Using the raw geohash as a LGBM native categorical allows the model to jointly learn location × time-of-day interactions directly.

### Temporal features
| Feature | Construction | Rationale |
|---------|-------------|-----------|
| `tmin` | `H * 60 + M` from timestamp string | Minutes within a day; test range (135–825) fully within training range (0–1425) |
| `hour` | `tmin // 60` | Coarser discretisation of time |
| `sin1`, `cos1` | `sin/cos(2π × tmin / 1440)` | Fundamental daily harmonic |
| `sin2`, `cos2` | `sin/cos(4π × tmin / 1440)` | Semi-diurnal harmonic |
| `sin3`, `cos3` | `sin/cos(6π × tmin / 1440)` | Tri-diurnal harmonic |

Cyclic harmonics give the model a smooth, continuous encoding of time-of-day that generalises across all 1440 minute positions, without the boundary artefacts of binning.

**Why `tmin` instead of `abs_time`:**  
`abs_time` is out-of-distribution for every test row (see Section 2). `tmin` encodes the position within a day — test values (135–825) fall inside the training range (0–1425) and the model generalises correctly.

### Road & environment features
| Feature | Construction |
|---------|-------------|
| `NumberofLanes` | Cast to float |
| `Temperature` | Median-imputed; `Temp_missing` flag added |
| `RoadType` | LGBM categorical; NaN → `"Missing"` |
| `LargeVehicles` | LGBM categorical; NaN → `"Missing"` |
| `Landmarks` | LGBM categorical; NaN → `"Missing"` |
| `Weather` | LGBM categorical; NaN → `"Missing"` |

### Lag feature (most impactful single feature)
| Feature | Construction | Coverage |
|---------|-------------|---------|
| `d48_demand` | Day-48 demand at the same (geohash, tmin) | 88.9% of test rows |

Day 48 is a full 24-hour observation of every geohash. For any test row at (geohash G, tmin T), the day-48 demand at (G, T) is the single strongest predictor — same location, same time of day, one day earlier.

**Construction details:**
- Built a lookup table `d48_demand[(geohash, tmin)]` from day-48 training rows.
- For day-49 training rows: look up the value and use it (valid — no leakage, day-48 data is earlier).
- For day-48 training rows: set to `NaN` (avoids self-reference / leakage).
- For the 11.1% of test rows with no day-48 match: `NaN` — LightGBM's built-in NaN routing assigns these to the statistically optimal split direction.

**Total features used: 20**
- 14 numeric: `lat`, `lon`, `tmin`, `hour`, `NumberofLanes`, `Temperature`, `Temp_missing`, `sin1–3`, `cos1–3`, `d48_demand`
- 5 categorical: `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`, `gh6`

---

## 4. Model Architecture

### Final model: LightGBM with 3-seed bagging

```python
LGBMRegressor(
    n_estimators     = 1000,
    learning_rate    = 0.03,
    num_leaves       = 127,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    min_child_samples = 10,
    n_jobs           = -1,
)
```

**Why LightGBM:**
- Native support for categorical features — handles `gh6` (1,249 levels) without ordinal encoding.
- Leaf-wise (best-first) tree growth suits the skewed, high-cardinality target distribution.
- Fast enough to run 3 × full-train fits on an M1 Pro in under 5 minutes.
- Built-in NaN routing handles missing `d48_demand` values optimally.

**Why 3-seed bagging:**  
Each seed produces slightly different subsampling and feature-fraction draws, decorrelating prediction errors. Averaging three predictions reduces variance without introducing bias. Experiments with 5+ seeds degraded LB performance, likely because additional diversity was not beneficial at this dataset size and tree depth.

```python
SEEDS = [42, 123, 2024]
final_pred = mean([clip(model(seed=s).predict(test), 0, 1) for s in SEEDS])
```

---

## 5. Validation Strategy

**3-fold cross-validation** with `KFold(n_splits=3, shuffle=True, random_state=42)` over the full training set.

The competition test set is a *future* window of the same smooth surface that training densely covers. Random K-fold (not a day-holdout) mirrors the real generalization gap because:
- Both train and test geohashes are drawn from the same ~1,249-location pool.
- The demand surface is smooth and stationary day-over-day (day 48 → day 49 is a small shift).
- A random fold held out from day 48 is structurally similar to the test set.

**OOF R²: 0.959** | **LB Score: 91.50%** | **Gap: ~4.5 pp** (gap is consistent and expected given the day-level shift).

---

## 6. Experiment History & What Failed

| Version | Key change | OOF R² | LB Score | Verdict |
|---------|-----------|--------|----------|---------|
| v1 | Baseline LGBM, static features only (lat/lon + harmonics) | 0.9081 | ~90.8% | Baseline |
| v2 | + Geohash target encoding (smoothed mean) | 0.9487 | — | OOF improved |
| v3 | Deeper LGBM (2000 trees, lr=0.02, L127) | 0.9505 | — | OOF improved |
| v4 | + XGBoost, CatBoost, ExtraTrees ensemble | 0.9511–0.9531 | **89.0%** | LB dropped — TE bias |
| v5 | Ridge stack of all 5 models | 0.9531 | 89.0% | Best OOF, worst LB |
| v6 | + `abs_time` feature | 0.950+ | **87.1%** | OOD disaster on LB |
| v7 | Removed `abs_time`, kept TE | — | ~89% | Still TE bias |
| v8 | Geohash as categorical, removed TE, + lag | 0.955 | ~91.0% | Big LB recovery |
| **v9** | **3-seed bagging of v8** | **0.959** | **91.50%** | **Final submission** |

### Failed approaches in detail

**Geohash target encoding (v2–v7):**  
Encoding each geohash as its all-day average demand (smoothed) is a strong OOF feature — OOF R² jumped from 0.908 to 0.949. But the test set covers only 02:15–13:45, which has systematically higher demand than the 24-hour average (morning peak hours). The encoding underestimates test demand, costing LB points despite looking strong in CV.

**`abs_time` (v6):**  
Added as `day * 1440 + tmin` to capture the day-level trend. OOF looked fine because day 48 and day 49 (00:00–02:00) in training span a reasonable range. But all 41,778 test rows have `abs_time` values (69255–69945) that exceed the training maximum (69240) — every test row falls into the same rightmost leaf. LB score collapsed to 87.1%.

**5-seed bagging:**  
Testing seeds beyond 3 (42, 123, 2024, 7, 999) consistently degraded LB by ~0.1–0.2 pp. Three seeds appear to be the sweet spot for this problem size.

**Geohash × hour target encoding:**  
Tested a finer (geohash, hour) encoding to capture within-location time-of-day patterns. Too few samples per cell (median ~7) made it noisy and overfit. OOF R² dropped from 0.949 to 0.942; rejected.

**MLP (PyTorch, MPS):**  
A 3-layer MLP (256→128→64) on standardised features achieved OOF R² 0.9357 — useful for ensemble diversity but individually weaker than tree models. Including it in ensembles consistently hurt or gave marginal gains; excluded from the final model.

---

## 7. Final Pipeline

```
dataset/train.csv
dataset/test.csv
        │
        ▼
1. Parse timestamp → tmin (minutes of day)
        │
        ▼
2. Decode geohash → (lat, lon) via base-32
        │
        ▼
3. Build features:
   - Cyclic harmonics: sin/cos at 1×, 2×, 3× daily frequency
   - Temperature: median impute + missing flag
   - Categoricals: RoadType, LargeVehicles, Landmarks, Weather → LGBM native
   - gh6: raw geohash string → LGBM native categorical
        │
        ▼
4. Add lag feature d48_demand:
   - Lookup day-48 demand at (geohash, tmin)
   - Day-48 train rows → NaN (no self-reference)
        │
        ▼
5. 3-fold OOF validation → OOF R² 0.959
        │
        ▼
6. 3-seed full-train LGBM (seeds 42, 123, 2024)
   → average predictions, clip to [0, 1]
        │
        ▼
submission_final.csv  (≡ submission_v9_bag.csv)
LB score: 91.50
```

---

## 8. Tools & Environment

| Tool | Version | Use |
|------|---------|-----|
| Python | 3.11 | Runtime |
| LightGBM | 4.6.0 | Final model |
| XGBoost | 3.0.0 | Experimentation only |
| CatBoost | latest | Experimentation only |
| scikit-learn | 1.x | KFold, metrics |
| pandas | 2.x | Data loading, feature construction |
| NumPy | 1.x | Numeric ops, harmonic features |
| PyTorch (MPS) | 2.x | MLP experimentation (not in final) |

**Hardware:** MacBook M1 Pro — all tree models use `n_jobs=-1` (CPU). No GPU required for the final model.

---

## 9. How to Reproduce

1. Place `train.csv` and `test.csv` in a `dataset/` subdirectory.
2. Install dependencies:
   ```bash
   pip install lightgbm scikit-learn pandas numpy
   ```
3. Open and run **`solution_final_v9.ipynb`** top to bottom.  
   The notebook produces `submission_final.csv` — identical to `submission_v9_bag.csv`.

**Expected runtime:** ~3–5 minutes on a modern CPU (M1 Pro or equivalent).

**Expected output:**
```
3-fold OOF R²: 0.9590
  seed  42: mean=0.0932
  seed 123: mean=0.0933
  seed 2024: mean=0.0931
submission_final.csv saved — shape=(41778, 2)
ACTUAL LB score: 91.50
```
# FlipKart-GridLock-2.0-ML-Challenge-
