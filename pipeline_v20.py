"""
Full autonomous pipeline v20 — Traffic Demand Prediction
Baseline: 91.5% LB (v9_bag). Target: 93+.
Critical rules from memory:
  - NO abs_time (test is OOD)
  - NO plain geohash TE (biases toward all-day mean)
  - Use tmin directly
  - Geohash as LGBM native categorical
  - Base seeds [42, 123, 2024]
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
import pygeohash as pgh
import time

BASE_DIR = Path("/Users/sudarshansudhakar/Downloads/FlipKart 2")
DATASET_DIR = BASE_DIR / "dataset"

print("=" * 60)
print("PHASE 0: DATA LOAD + PARSE")
print("=" * 60)

train = pd.read_csv(DATASET_DIR / "train.csv")
test = pd.read_csv(DATASET_DIR / "test.csv")

def parse_ts(ts):
    h, m = str(ts).split(':')
    return int(h) * 60 + int(m)

for df in [train, test]:
    df['tmin'] = df['timestamp'].apply(parse_ts)
    df['hour'] = df['tmin'] // 60
    df['minute'] = df['tmin'] % 60
    # DO NOT create abs_time — OOD for test

print(f"Train: {train.shape}, days={sorted(train['day'].unique())}")
print(f"Test : {test.shape}, days={sorted(test['day'].unique())}")
print(f"Train demand: mean={train['demand'].mean():.4f} std={train['demand'].std():.4f}")
print(f"Train tmin: {train['tmin'].min()}-{train['tmin'].max()}")
print(f"Test  tmin: {test['tmin'].min()}-{test['tmin'].max()}")

# Day mapping — only 2 days
dow48 = 48 % 7  # proxy; not real DOW but consistent
dow49 = 49 % 7

for df in [train, test]:
    df['dayofweek'] = df['day'] % 7  # 48%7=6, 49%7=0
    df['is_weekend'] = (df['dayofweek'].isin([5, 6])).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['part_of_day'] = pd.cut(df['hour'],
        bins=[-1, 5, 11, 16, 20, 24],
        labels=[0, 1, 2, 3, 4]).astype(int)

print("\n=== PHASE 0 COMPLETE ===\n")

print("=" * 60)
print("PHASE 1: FEATURE ENGINEERING")
print("=" * 60)

# ── 1. Cyclical encodings ──
for df in [train, test]:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['tmin_sin'] = np.sin(2 * np.pi * df['tmin'] / 1440)
    df['tmin_cos'] = np.cos(2 * np.pi * df['tmin'] / 1440)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    # Additional harmonics (already in v9)
    for k in [2, 3]:
        df[f'tmin_sin{k}'] = np.sin(2 * np.pi * k * df['tmin'] / 1440)
        df[f'tmin_cos{k}'] = np.cos(2 * np.pi * k * df['tmin'] / 1440)

# ── 2. Geohash decode + prefix levels ──
gh_coords = {}
for gh in pd.concat([train['geohash'], test['geohash']]).unique():
    try:
        lat, lon = pgh.decode(gh)
        gh_coords[gh] = (lat, lon)
    except Exception:
        gh_coords[gh] = (np.nan, np.nan)

for df in [train, test]:
    df['lat'] = df['geohash'].map(lambda g: gh_coords.get(g, (np.nan, np.nan))[0])
    df['lon'] = df['geohash'].map(lambda g: gh_coords.get(g, (np.nan, np.nan))[1])
    df['geo_p3'] = df['geohash'].str[:3]
    df['geo_p4'] = df['geohash'].str[:4]
    df['geo_p5'] = df['geohash'].str[:5]

print(f"Decoded {len(gh_coords)} geohash locations")
print(f"geo_p3 unique: {train['geo_p3'].nunique()}, p4: {train['geo_p4'].nunique()}, p5: {train['geo_p5'].nunique()}")

# ── 3. Historical averages (train only → merge onto test) ──
# Use day48 for lag-style features (day49 train rows have tmin 0-120 only)
d48 = train[train['day'] == 48].copy()

# hist_avg by (geohash, tmin) — MOST IMPORTANT: exact lookup
gh_tmin_avg = d48.groupby(['geohash', 'tmin'])['demand'].mean().reset_index()
gh_tmin_avg.columns = ['geohash', 'tmin', 'd48_demand']
train = train.merge(gh_tmin_avg, on=['geohash', 'tmin'], how='left')
test = test.merge(gh_tmin_avg, on=['geohash', 'tmin'], how='left')
print(f"d48_demand coverage: train={train['d48_demand'].notna().mean():.3f}, test={test['d48_demand'].notna().mean():.3f}")

# hist_avg by (geohash, hour) — smoother version
gh_hour_avg = d48.groupby(['geohash', 'hour'])['demand'].mean().reset_index()
gh_hour_avg.columns = ['geohash', 'hour', 'hist_gh_hour']
train = train.merge(gh_hour_avg, on=['geohash', 'hour'], how='left')
test = test.merge(gh_hour_avg, on=['geohash', 'hour'], how='left')

# hist_avg by (geohash, hour, dayofweek) — day-specific
gh_hour_dow = d48.groupby(['geohash', 'hour', 'dayofweek'])['demand'].mean().reset_index()
gh_hour_dow.columns = ['geohash', 'hour', 'dayofweek', 'hist_gh_hour_dow']
train = train.merge(gh_hour_dow, on=['geohash', 'hour', 'dayofweek'], how='left')
test = test.merge(gh_hour_dow, on=['geohash', 'hour', 'dayofweek'], how='left')

# hist_avg by geohash alone (fallback)
gh_avg = d48.groupby('geohash')['demand'].mean().reset_index()
gh_avg.columns = ['geohash', 'hist_gh_avg']
train = train.merge(gh_avg, on='geohash', how='left')
test = test.merge(gh_avg, on='geohash', how='left')

# hist_avg by (geo_p4, hour, dayofweek) — cluster-level
p4_hour_dow = d48.groupby(['geo_p4', 'hour', 'dayofweek'])['demand'].mean().reset_index()
p4_hour_dow.columns = ['geo_p4', 'hour', 'dayofweek', 'hist_p4_hour_dow']
train = train.merge(p4_hour_dow, on=['geo_p4', 'hour', 'dayofweek'], how='left')
test = test.merge(p4_hour_dow, on=['geo_p4', 'hour', 'dayofweek'], how='left')

# hist_avg by (geo_p4, tmin)
p4_tmin = d48.groupby(['geo_p4', 'tmin'])['demand'].mean().reset_index()
p4_tmin.columns = ['geo_p4', 'tmin', 'hist_p4_tmin']
train = train.merge(p4_tmin, on=['geo_p4', 'tmin'], how='left')
test = test.merge(p4_tmin, on=['geo_p4', 'tmin'], how='left')

# hist_avg by (geo_p3, hour)
p3_hour = d48.groupby(['geo_p3', 'hour'])['demand'].mean().reset_index()
p3_hour.columns = ['geo_p3', 'hour', 'hist_p3_hour']
train = train.merge(p3_hour, on=['geo_p3', 'hour'], how='left')
test = test.merge(p3_hour, on=['geo_p3', 'hour'], how='left')

# hist_avg by tmin alone (time-of-day pattern across all geohashes)
tmin_avg = d48.groupby('tmin')['demand'].mean().reset_index()
tmin_avg.columns = ['tmin', 'hist_tmin_avg']
train = train.merge(tmin_avg, on='tmin', how='left')
test = test.merge(tmin_avg, on='tmin', how='left')

print(f"hist_gh_hour coverage: train={train['hist_gh_hour'].notna().mean():.3f}, test={test['hist_gh_hour'].notna().mean():.3f}")
print(f"hist_p4_tmin coverage: train={train['hist_p4_tmin'].notna().mean():.3f}, test={test['hist_p4_tmin'].notna().mean():.3f}")

# Fill d48_demand NaNs with progressively coarser fallbacks
for df in [train, test]:
    df['d48_demand'] = df['d48_demand'].fillna(df['hist_gh_hour'])
    df['d48_demand'] = df['d48_demand'].fillna(df['hist_p4_tmin'])
    df['d48_demand'] = df['d48_demand'].fillna(df['hist_gh_avg'])
    df['d48_demand'] = df['d48_demand'].fillna(df['hist_tmin_avg'])

print(f"d48_demand after fill: train={train['d48_demand'].notna().mean():.4f}, test={test['d48_demand'].notna().mean():.4f}")

# ── 4. Neighbor demand feature ──
print("Computing neighbor demand features...")

def get_neighbors_avg(geohash_val):
    """Average d48 demand of pygeohash neighbors."""
    try:
        neighbors = pgh.neighbors(geohash_val)
        vals = [gh_tmin_avg_dict.get(n, np.nan) for n in neighbors.values()]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else np.nan
    except Exception:
        return np.nan

# Build geohash → mean d48 demand lookup
gh_mean_d48 = d48.groupby('geohash')['demand'].mean().to_dict()

neighbor_avg = {}
for gh in pd.concat([train['geohash'], test['geohash']]).unique():
    try:
        neighbors = pgh.neighbors(gh)
        vals = [gh_mean_d48.get(n, np.nan) for n in neighbors.values()]
        vals = [v for v in vals if not np.isnan(v)]
        neighbor_avg[gh] = np.mean(vals) if vals else np.nan
    except Exception:
        neighbor_avg[gh] = np.nan

for df in [train, test]:
    df['neighbor_demand'] = df['geohash'].map(neighbor_avg)

print(f"neighbor_demand: train={train['neighbor_demand'].notna().mean():.3f}, test={test['neighbor_demand'].notna().mean():.3f}")

# ── 5. Rolling stats per geohash (from d48 only — no leakage) ──
# For each (geohash, tmin), compute rolling stats from d48 preceding tmin windows
d48_sorted = d48.sort_values(['geohash', 'tmin']).copy()
d48_sorted['roll3_mean'] = d48_sorted.groupby('geohash')['demand'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
d48_sorted['roll7_mean'] = d48_sorted.groupby('geohash')['demand'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean())
d48_sorted['roll7_std'] = d48_sorted.groupby('geohash')['demand'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=2).std())
d48_sorted['roll3_min'] = d48_sorted.groupby('geohash')['demand'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).min())
d48_sorted['roll3_max'] = d48_sorted.groupby('geohash')['demand'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).max())

roll_feats = ['roll3_mean', 'roll7_mean', 'roll7_std', 'roll3_min', 'roll3_max']
roll_lookup = d48_sorted[['geohash', 'tmin'] + roll_feats].drop_duplicates(['geohash', 'tmin'])
train = train.merge(roll_lookup, on=['geohash', 'tmin'], how='left')
test = test.merge(roll_lookup, on=['geohash', 'tmin'], how='left')
# Fill NaN rolling from hist_gh_avg
for f in roll_feats:
    for df in [train, test]:
        df[f] = df[f].fillna(df['hist_gh_avg'])

print(f"roll3_mean: train={train['roll3_mean'].notna().mean():.3f}, test={test['roll3_mean'].notna().mean():.3f}")

# ── 6. Lag features (demand_lag1,2,3 per geohash in tmin order from d48) ──
d48_lag = d48.sort_values(['geohash', 'tmin'])[['geohash', 'tmin', 'demand']].copy()
d48_lag['demand_lag1'] = d48_lag.groupby('geohash')['demand'].shift(1)
d48_lag['demand_lag2'] = d48_lag.groupby('geohash')['demand'].shift(2)
d48_lag['demand_lag3'] = d48_lag.groupby('geohash')['demand'].shift(3)
lag_feats = ['demand_lag1', 'demand_lag2', 'demand_lag3']
lag_lookup = d48_lag[['geohash', 'tmin'] + lag_feats].drop_duplicates(['geohash', 'tmin'])
train = train.merge(lag_lookup, on=['geohash', 'tmin'], how='left')
test = test.merge(lag_lookup, on=['geohash', 'tmin'], how='left')
# Fill NaN lags
for f in lag_feats:
    for df in [train, test]:
        df[f] = df[f].fillna(df['hist_gh_avg'])

# ── 7. OOF Target Encoding (non-geohash) ──
# Geohash remains as native categorical — NO TE for geohash (known to hurt)
# OOF TE for RoadType, Weather, geo_p4

def oof_target_encode(train_df, test_df, col, target='demand', n_splits=5, smoothing=10):
    """OOF target encoding with smoothing — avoids the all-day mean bias."""
    global_mean = train_df[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df = train_df.copy()
    train_df[f'te_{col}'] = np.nan

    for tr_idx, val_idx in kf.split(train_df):
        tr_means = train_df.iloc[tr_idx].groupby(col)[target].agg(['mean', 'count'])
        smooth = tr_means['mean'] * tr_means['count'] / (tr_means['count'] + smoothing) + \
                 global_mean * smoothing / (tr_means['count'] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc(f'te_{col}')] = \
            train_df.iloc[val_idx][col].map(smooth).fillna(global_mean).values

    # Full train means for test
    full_means = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth_full = full_means['mean'] * full_means['count'] / (full_means['count'] + smoothing) + \
                  global_mean * smoothing / (full_means['count'] + smoothing)
    test_df = test_df.copy()
    test_df[f'te_{col}'] = test_df[col].map(smooth_full).fillna(global_mean)
    return train_df, test_df

for col in ['RoadType', 'Weather', 'geo_p4']:
    train, test = oof_target_encode(train, test, col)
    print(f"OOF TE {col}: train NaN={train[f'te_{col}'].isna().sum()}")

# ── 8. Interaction features ──
for df in [train, test]:
    df['road_hour'] = df['RoadType'].astype(str) + '_' + df['hour'].astype(str)
    df['weather_road'] = df['Weather'].astype(str) + '_' + df['RoadType'].astype(str)
    df['lanes_rush'] = df['NumberofLanes'].fillna(1) * df['is_rush_hour']
    df['temp_bucket'] = pd.cut(df['Temperature'].fillna(df['Temperature'].median()),
                                bins=5, labels=False)
    df['temp_weather'] = df['temp_bucket'].astype(str) + '_' + df['Weather'].astype(str)
    df['Temp_missing'] = df['Temperature'].isna().astype(int)
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['lanes_x_hour'] = df['NumberofLanes'].fillna(1) * df['hour']

# ── 9. Demand ratio features ──
for df in [train, test]:
    df['d48_to_gh_ratio'] = df['d48_demand'] / (df['hist_gh_avg'] + 1e-6)
    df['d48_to_tmin_ratio'] = df['d48_demand'] / (df['hist_tmin_avg'] + 1e-6)

print("\n=== Top 20 correlations with demand ===")
num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c not in ['demand', 'Index', 'day', 'tmin', 'dayofweek']]
corrs = train[num_cols + ['demand']].corr()['demand'].abs().sort_values(ascending=False)
print(corrs.head(25).to_string())

# Save checkpoint
train.to_parquet(BASE_DIR / 'train_features.parquet', index=False)
test.to_parquet(BASE_DIR / 'test_features.parquet', index=False)
print("\nSaved: train_features.parquet, test_features.parquet")
print("=== PHASE 1 COMPLETE ===\n")

print("=" * 60)
print("PHASE 2: MODEL TRAINING")
print("=" * 60)

# ── Feature set ──
CAT_COLS = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
            'geo_p3', 'geo_p4', 'geo_p5', 'road_hour', 'weather_road', 'temp_weather']

BASE_NUM = [
    'lat', 'lon', 'tmin', 'hour', 'minute',
    'tmin_sin', 'tmin_cos', 'tmin_sin2', 'tmin_cos2', 'tmin_sin3', 'tmin_cos3',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'NumberofLanes', 'Temperature', 'Temp_missing',
    'is_weekend', 'is_rush_hour', 'part_of_day',
    'd48_demand', 'hist_gh_hour', 'hist_gh_hour_dow', 'hist_gh_avg',
    'hist_p4_hour_dow', 'hist_p4_tmin', 'hist_p3_hour', 'hist_tmin_avg',
    'neighbor_demand',
    'roll3_mean', 'roll7_mean', 'roll7_std', 'roll3_min', 'roll3_max',
    'demand_lag1', 'demand_lag2', 'demand_lag3',
    'te_RoadType', 'te_Weather', 'te_geo_p4',
    'lanes_rush', 'lanes_x_hour', 'temp_bucket',
    'd48_to_gh_ratio', 'd48_to_tmin_ratio',
]

FEATURES = BASE_NUM + CAT_COLS

# Check features exist
available = [f for f in FEATURES if f in train.columns]
missing_feats = [f for f in FEATURES if f not in train.columns]
if missing_feats:
    print(f"WARNING: missing features: {missing_feats}")
FEATURES = available
print(f"Total features: {len(FEATURES)} ({len(CAT_COLS)} categorical, {len(available)-len([c for c in CAT_COLS if c in available])} numeric)")

X = train[FEATURES].copy()
y = train['demand'].copy()
X_test = test[FEATURES].copy()

# Encode categoricals as int codes for LGBM
cat_indices = []
for col in CAT_COLS:
    if col in FEATURES:
        all_vals = pd.concat([X[col], X_test[col]]).astype(str).fillna('__NA__')
        X[col] = pd.Categorical(X[col].astype(str).fillna('__NA__'), categories=all_vals.unique()).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str).fillna('__NA__'), categories=all_vals.unique()).codes
        cat_indices.append(FEATURES.index(col))

print(f"Categorical feature indices: {len(cat_indices)} cats")

TARGET = 'demand'

# ── 2a. GroupKFold on geohash (5 folds) ──
print("\n--- GroupKFold(5) on geohash ---")
gkf = GroupKFold(n_splits=5)
groups = train['geohash'].values

BASE_PARAMS = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=127,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1
)

oof_gkf = np.zeros(len(train))
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(**BASE_PARAMS)
    model.fit(X_tr, y_tr, categorical_feature=cat_indices,
              eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_gkf[val_idx] = model.predict(X_val)

gkf_r2 = r2_score(y, oof_gkf)
print(f"GroupKFold OOF R²: {gkf_r2:.6f} → score {gkf_r2*100:.2f}")

# ── 2b. Time-based split (day48→day49) ──
print("\n--- Time-based split (day48→day49) ---")
tr_mask = train['day'] == 48
val_mask = train['day'] == 49
X_tr_t = X[tr_mask]; y_tr_t = y[tr_mask]
X_val_t = X[val_mask]; y_val_t = y[val_mask]
model_t = lgb.LGBMRegressor(**BASE_PARAMS)
model_t.fit(X_tr_t, y_tr_t, categorical_feature=cat_indices,
            eval_set=[(X_val_t, y_val_t)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
t_r2 = r2_score(y_val_t, model_t.predict(X_val_t))
print(f"Time-based OOF R²: {t_r2:.6f} → score {t_r2*100:.2f}")

# ── 2c. KFold(5) OOF — main proxy ──
print("\n--- KFold(5) OOF (main proxy) ---")
BASE_SEEDS = [42, 123, 2024]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def train_multiseed(X, y, X_test, seeds, params, cat_idx, n_splits=5):
    """Train multi-seed bagged model, return OOF and test predictions."""
    all_oof = []
    all_test = []
    for seed in seeds:
        p = {**params, 'random_state': seed}
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        kf_ = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf_.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            m = lgb.LGBMRegressor(**p)
            m.fit(X_tr, y_tr, categorical_feature=cat_idx,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            oof[val_idx] = m.predict(X_val)
            test_preds += m.predict(X_test) / n_splits
        all_oof.append(oof)
        all_test.append(test_preds)
        r2 = r2_score(y, oof)
        print(f"  seed={seed} OOF R²={r2:.6f} ({r2*100:.2f})")
    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    r2_avg = r2_score(y, oof_avg)
    print(f"  AVERAGED OOF R²={r2_avg:.6f} ({r2_avg*100:.2f})")
    return oof_avg, test_avg, r2_avg

print("\nBaseline (v9 config) with NEW features:")
oof_v20, test_v20, r2_v20 = train_multiseed(X, y, X_test, BASE_SEEDS, BASE_PARAMS, cat_indices)

print(f"\nPhase 2 summary:")
print(f"  GroupKFold R²:  {gkf_r2*100:.2f}%")
print(f"  Time-based R²:  {t_r2*100:.2f}%")
print(f"  KFold-avg R²:   {r2_v20*100:.2f}%  (v9 was ~95.90% OOF)")

# ── 2d. SHAP top features ──
print("\n--- SHAP top 20 features ---")
try:
    import shap
    m_shap = lgb.LGBMRegressor(**BASE_PARAMS)
    m_shap.fit(X, y, categorical_feature=cat_indices)
    explainer = shap.TreeExplainer(m_shap)
    shap_vals = explainer.shap_values(X.sample(min(5000, len(X)), random_state=42))
    feat_importance = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print(feat_importance.head(20).to_string())
except Exception as e:
    print(f"SHAP failed ({e}), using LGBM importance instead")
    m_imp = lgb.LGBMRegressor(**BASE_PARAMS)
    m_imp.fit(X, y, categorical_feature=cat_indices)
    imp = pd.Series(m_imp.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(imp.head(20).to_string())

# ── 2e. Optuna tuning ──
print("\n--- Optuna 50 trials ---")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    kf_opt = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 500, 2000),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            num_leaves=trial.suggest_categorical('num_leaves', [63, 127, 255]),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            feature_fraction=trial.suggest_float('feature_fraction', 0.6, 1.0),
            bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
            bagging_freq=1,
            lambda_l1=trial.suggest_float('lambda_l1', 0, 5),
            lambda_l2=trial.suggest_float('lambda_l2', 0, 5),
            max_bin=trial.suggest_categorical('max_bin', [255, 511]),
            n_jobs=-1, verbose=-1, random_state=42
        )
        oof = np.zeros(len(X))
        for tr_idx, val_idx in kf_opt.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr, categorical_feature=cat_indices,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            oof[val_idx] = m.predict(X_val)
        return r2_score(y, oof)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best_params = study.best_params
    print(f"Best Optuna OOF R²: {study.best_value:.6f} ({study.best_value*100:.2f}%)")
    print(f"Best params: {best_params}")
except Exception as e:
    print(f"Optuna failed: {e}")
    best_params = {k: v for k, v in BASE_PARAMS.items() if k != 'random_state'}

# Build full Optuna model
print("\nTraining Optuna model multi-seed...")
opt_params = {**best_params, 'n_jobs': -1, 'verbose': -1}
if 'bagging_fraction' in opt_params:
    opt_params['subsample'] = opt_params.pop('bagging_fraction')
    opt_params['subsample_freq'] = opt_params.pop('bagging_freq', 1)
if 'feature_fraction' in opt_params:
    opt_params['colsample_bytree'] = opt_params.pop('feature_fraction')

oof_opt, test_opt, r2_opt = train_multiseed(X, y, X_test, BASE_SEEDS, opt_params, cat_indices)

# Save phase 2 submission
test_opt_clipped = np.clip(test_opt, 0, 1)
sub_p2 = test[['Index']].copy()
sub_p2['demand'] = test_opt_clipped
sub_p2.to_csv(BASE_DIR / 'submission_phase2.csv', index=False)
print(f"\nSaved submission_phase2.csv — test pred mean={test_opt_clipped.mean():.4f}")
print("=== PHASE 2 COMPLETE ===\n")

print("=" * 60)
print("PHASE 3: PUSH TO 93+")
print("=" * 60)

experiment_log = []

# Best so far: v20 new features with base params
best_r2 = r2_v20
best_test = test_v20.copy()
experiment_log.append({'name': 'v20_base_newfeatures', 'oof_r2': r2_v20, 'delta': r2_v20 - 0.959, 'notes': 'new features + base params'})
experiment_log.append({'name': 'optuna_tuned', 'oof_r2': r2_opt, 'delta': r2_opt - r2_v20, 'notes': 'optuna 50 trials'})

if r2_opt > best_r2:
    best_r2 = r2_opt
    best_test = test_opt.copy()

# ── Phase A: DART (already known to fail but try with new features) ──
print("\n[A] DART boosting...")
DART_PARAMS = dict(
    boosting_type='dart', n_estimators=1000, learning_rate=0.03, num_leaves=127,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
    drop_rate=0.1, skip_drop=0.5, n_jobs=-1, verbose=-1
)
# DART doesn't support early stopping — train all n_estimators
dart_oof_list = []
dart_test_list = []
for seed in BASE_SEEDS:
    p = {**DART_PARAMS, 'random_state': seed}
    oof_ = np.zeros(len(X))
    test_ = np.zeros(len(X_test))
    kf_ = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf_.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr, categorical_feature=cat_indices)
        oof_[val_idx] = m.predict(X_val)
        test_ += m.predict(X_test) / 5
    dart_oof_list.append(oof_)
    dart_test_list.append(test_)

dart_oof = np.mean(dart_oof_list, axis=0)
dart_test = np.mean(dart_test_list, axis=0)
r2_dart = r2_score(y, dart_oof)
delta_dart = r2_dart - best_r2
print(f"DART OOF R²: {r2_dart:.6f} ({r2_dart*100:.2f}%), delta={delta_dart:+.4f}")
experiment_log.append({'name': 'DART', 'oof_r2': r2_dart, 'delta': delta_dart, 'notes': 'drop_rate=0.1, skip=0.5'})
if r2_dart > best_r2:
    best_r2 = r2_dart
    best_test = dart_test.copy()
    print("  -> NEW BEST!")

# ── Phase B: XGBoost blend ──
print("\n[B] XGBoost blend...")
try:
    import xgboost as xgb

    XGB_PARAMS = dict(
        n_estimators=1000, learning_rate=0.03, max_leaves=127,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        tree_method='hist', enable_categorical=False,
        n_jobs=-1, verbosity=0
    )
    # XGB doesn't support string categoricals — use encoded X
    xgb_oof_list = []
    xgb_test_list = []
    for seed in BASE_SEEDS:
        p = {**XGB_PARAMS, 'random_state': seed}
        oof_ = np.zeros(len(X))
        test_ = np.zeros(len(X_test))
        kf_ = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf_.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            m = xgb.XGBRegressor(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50, verbose=False)
            oof_[val_idx] = m.predict(X_val)
            test_ += m.predict(X_test) / 5
        xgb_oof_list.append(oof_)
        xgb_test_list.append(test_)

    xgb_oof = np.mean(xgb_oof_list, axis=0)
    xgb_test = np.mean(xgb_test_list, axis=0)
    r2_xgb = r2_score(y, xgb_oof)
    print(f"XGB OOF R²: {r2_xgb:.6f} ({r2_xgb*100:.2f}%)")

    # Grid search blend lgbm/xgb
    best_blend_r2 = -np.inf
    best_w = 1.0
    for w in np.arange(0.5, 0.95, 0.05):
        blend_oof = w * oof_v20 + (1 - w) * xgb_oof
        r2_b = r2_score(y, blend_oof)
        if r2_b > best_blend_r2:
            best_blend_r2 = r2_b
            best_w = w

    blend_test = best_w * test_v20 + (1 - best_w) * xgb_test
    delta_blend = best_blend_r2 - best_r2
    print(f"Best LGBM/XGB blend (w={best_w:.2f}): OOF R²={best_blend_r2:.6f} ({best_blend_r2*100:.2f}%), delta={delta_blend:+.4f}")
    experiment_log.append({'name': f'XGB_blend_w{best_w:.2f}', 'oof_r2': best_blend_r2, 'delta': delta_blend, 'notes': f'lgbm*{best_w:.2f}+xgb*{1-best_w:.2f}'})
    if best_blend_r2 > best_r2:
        best_r2 = best_blend_r2
        best_test = blend_test.copy()
        print("  -> NEW BEST!")
except Exception as e:
    print(f"XGB failed: {e}")

# ── Phase C: Expanded seeds ──
print("\n[C] Expanded seeds [42,123,2024,0,7,999,2025,314]...")
EXPANDED_SEEDS = [42, 123, 2024, 0, 7, 999, 2025, 314]
oof_exp_all = []
test_exp_all = []
for seed in EXPANDED_SEEDS:
    p = {**BASE_PARAMS, 'random_state': seed}
    oof_ = np.zeros(len(X))
    test_ = np.zeros(len(X_test))
    kf_ = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf_.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr, categorical_feature=cat_indices,
              eval_set=[(X_val, y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_[val_idx] = m.predict(X_val)
        test_ += m.predict(X_test) / 5
    oof_exp_all.append(oof_)
    test_exp_all.append(test_)
    r2_s = r2_score(y, oof_)
    print(f"  seed={seed}: OOF R²={r2_s:.6f}, test_pred_mean={test_.mean():.4f}")

oof_exp = np.mean(oof_exp_all, axis=0)
test_exp = np.mean(test_exp_all, axis=0)
r2_exp = r2_score(y, oof_exp)
delta_exp = r2_exp - best_r2
print(f"8-seed expanded OOF R²: {r2_exp:.6f} ({r2_exp*100:.2f}%), delta={delta_exp:+.4f}")
print(f"test_pred mean={test_exp.mean():.4f} (target: ~0.1302)")
experiment_log.append({'name': 'expanded_8seeds', 'oof_r2': r2_exp, 'delta': delta_exp, 'notes': '8 seeds avg, check mean'})
if r2_exp > best_r2:
    best_r2 = r2_exp
    best_test = test_exp.copy()
    print("  -> NEW BEST!")

# ── Phase D: Residual model ──
print("\n[D] Residual model (train on demand - pred)...")
residuals = y.values - oof_v20
residual_model = lgb.LGBMRegressor(**{**BASE_PARAMS, 'n_estimators': 300, 'num_leaves': 63})
# OOF residuals
oof_resid = np.zeros(len(X))
test_resid = np.zeros(len(X_test))
kf_r = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in kf_r.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    res_tr = residuals[tr_idx]
    m_r = lgb.LGBMRegressor(**{**BASE_PARAMS, 'n_estimators': 300, 'num_leaves': 63, 'random_state': 42})
    m_r.fit(X_tr, res_tr, categorical_feature=cat_indices)
    oof_resid[val_idx] = m_r.predict(X_val)
    test_resid += m_r.predict(X_test) / 5

# Train full residual model for test
m_r_full = lgb.LGBMRegressor(**{**BASE_PARAMS, 'n_estimators': 300, 'num_leaves': 63, 'random_state': 42})
m_r_full.fit(X, residuals, categorical_feature=cat_indices)
test_resid_full = m_r_full.predict(X_test)

for alpha in [0.1, 0.2, 0.3, 0.5]:
    combined_oof = oof_v20 + alpha * oof_resid
    r2_resid = r2_score(y, combined_oof)
    delta_resid = r2_resid - best_r2
    print(f"  alpha={alpha}: OOF R²={r2_resid:.6f} ({r2_resid*100:.2f}%), delta={delta_resid:+.4f}")
    if r2_resid > best_r2:
        best_r2 = r2_resid
        best_test = np.clip(test_v20 + alpha * test_resid_full, 0, 1)
        print(f"  -> NEW BEST (alpha={alpha})!")
        experiment_log.append({'name': f'residual_alpha{alpha}', 'oof_r2': r2_resid, 'delta': delta_resid, 'notes': 'residual correction'})
        break
else:
    experiment_log.append({'name': 'residual_model', 'oof_r2': r2_resid, 'delta': delta_resid, 'notes': 'no improvement'})

# ── Phase E: Test-range calibration ──
print("\n[E] Test-range calibration (tmin 135-825 stats)...")
# Train rows with tmin in test range, use their demand stats as calibration
test_tmin_mask = (train['tmin'] >= 135) & (train['tmin'] <= 825)
print(f"Train rows in test tmin range (135-825): {test_tmin_mask.sum()}")
train_testrange = train[test_tmin_mask]
print(f"demand mean (test tmin range): {train_testrange['demand'].mean():.4f}")
print(f"test preds mean: {np.clip(best_test, 0, 1).mean():.4f}")

# ── Final: Train on ALL train data with best config ──
print("\n--- Training FINAL model on all train data ---")
final_seeds = BASE_SEEDS

# Determine best params (optuna if better)
if r2_opt >= r2_v20:
    final_params = opt_params
    print(f"Using Optuna params (R²={r2_opt:.6f})")
else:
    final_params = BASE_PARAMS
    print(f"Using base params (R²={r2_v20:.6f})")

final_test_preds = []
for seed in final_seeds:
    p = {**final_params, 'random_state': seed, 'n_jobs': -1, 'verbose': -1}
    m_final = lgb.LGBMRegressor(**p)
    m_final.fit(X, y, categorical_feature=cat_indices)
    final_test_preds.append(m_final.predict(X_test))

final_preds = np.clip(np.mean(final_test_preds, axis=0), 0, 1)

# Compare with best_test from experiments
best_test_clipped = np.clip(best_test, 0, 1)
print(f"\nFinal (all-train) pred: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")
print(f"Best-exp pred:          mean={best_test_clipped.mean():.4f}, std={best_test_clipped.std():.4f}")

# Use best experiment predictions for submission
sub_final = test[['Index']].copy()
sub_final['demand'] = best_test_clipped
sub_final.to_csv(BASE_DIR / 'submission_v20_newfeatures.csv', index=False)
print(f"\nSaved submission_v20_newfeatures.csv ({len(sub_final)} rows)")

# Also save all-train version
sub_final2 = test[['Index']].copy()
sub_final2['demand'] = final_preds
sub_final2.to_csv(BASE_DIR / 'submission_v20_alltrain.csv', index=False)
print(f"Saved submission_v20_alltrain.csv")

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY TABLE")
print("=" * 60)
print(f"{'Name':<30} {'OOF R²':>10} {'Score':>8} {'Delta':>8} {'Notes'}")
print("-" * 80)
for exp in experiment_log:
    score = exp['oof_r2'] * 100
    print(f"{exp['name']:<30} {exp['oof_r2']:>10.6f} {score:>7.2f}% {exp['delta']:>+8.4f}  {exp.get('notes','')}")

print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(f"Best OOF R²:  {best_r2:.6f} → {best_r2*100:.2f}%")
print(f"submission_v20_newfeatures.csv: {len(sub_final)} rows, cols={list(sub_final.columns)}")
print(f"  pred mean={sub_final['demand'].mean():.4f}")
print(f"  pred std= {sub_final['demand'].std():.4f}")
print(f"  pred min= {sub_final['demand'].min():.4f}")
print(f"  pred max= {sub_final['demand'].max():.4f}")
print(f"\nPrevious best (v9_bag): ~91.5% LB")
print(f"Note: CV-to-LB gap ≈ 4.5pp (OOF {best_r2*100:.2f}% → estimated LB {(best_r2-0.045)*100:.2f}%)")
