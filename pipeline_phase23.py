"""
Pipeline Phase 2+3 — loads from parquet checkpoint, skips SHAP
"""
import warnings
warnings.filterwarnings('ignore')
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
import time

BASE_DIR = Path("/Users/sudarshansudhakar/Downloads/FlipKart 2")

print("Loading feature parquets...", flush=True)
train = pd.read_parquet(BASE_DIR / 'train_features_fixed.parquet')
test = pd.read_parquet(BASE_DIR / 'test_features_fixed.parquet')
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)
print(f"Train columns: {list(train.columns)}", flush=True)

y = train['demand'].copy()

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

FEATURES = [f for f in BASE_NUM + CAT_COLS if f in train.columns]
missing_feats = [f for f in BASE_NUM + CAT_COLS if f not in train.columns]
if missing_feats:
    print(f"Missing features (skipping): {missing_feats}", flush=True)
print(f"Using {len(FEATURES)} features ({len([c for c in CAT_COLS if c in FEATURES])} categorical)", flush=True)

X = train[FEATURES].copy()
X_test = test[FEATURES].copy()

# Encode categoricals
cat_indices = []
for col in CAT_COLS:
    if col in FEATURES:
        all_vals = pd.concat([X[col], X_test[col]]).astype(str).fillna('__NA__')
        X[col] = pd.Categorical(X[col].astype(str).fillna('__NA__'), categories=all_vals.unique()).codes
        X_test[col] = pd.Categorical(X_test[col].astype(str).fillna('__NA__'), categories=all_vals.unique()).codes
        cat_indices.append(FEATURES.index(col))

print(f"Categorical indices: {len(cat_indices)}", flush=True)

BASE_PARAMS = dict(
    n_estimators=1000, learning_rate=0.03, num_leaves=127,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1
)

BASE_SEEDS = [42, 123, 2024]

def train_multiseed(X, y, X_test, seeds, params, cat_idx, n_splits=5, tag=''):
    all_oof, all_test = [], []
    for seed in seeds:
        p = {**params, 'random_state': seed}
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        kf_ = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(kf_.split(X)):
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
        print(f"  {tag} seed={seed} OOF R²={r2:.6f} ({r2*100:.2f}%)", flush=True)
    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    r2_avg = r2_score(y, oof_avg)
    print(f"  {tag} AVERAGED R²={r2_avg:.6f} ({r2_avg*100:.2f}%)", flush=True)
    return oof_avg, test_avg, r2_avg

print("\n" + "="*60, flush=True)
print("PHASE 2: MODEL TRAINING", flush=True)
print("="*60, flush=True)

# ── GroupKFold ──
print("\n--- GroupKFold(5) on geohash ---", flush=True)
gkf = GroupKFold(n_splits=5)
groups = train['geohash'].values
oof_gkf = np.zeros(len(train))
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    m = lgb.LGBMRegressor(**BASE_PARAMS)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_indices,
          eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_gkf[val_idx] = m.predict(X.iloc[val_idx])
    print(f"  fold {fold}: R²={r2_score(y.iloc[val_idx], oof_gkf[val_idx]):.4f}", flush=True)
gkf_r2 = r2_score(y, oof_gkf)
print(f"GroupKFold OOF R²: {gkf_r2:.6f} ({gkf_r2*100:.2f}%)", flush=True)

# ── Time-based split ──
print("\n--- Time-based split (day48→day49) ---", flush=True)
tr_mask = train['day'] == 48
val_mask = train['day'] == 49
model_t = lgb.LGBMRegressor(**BASE_PARAMS)
model_t.fit(X[tr_mask], y[tr_mask], categorical_feature=cat_indices,
            eval_set=[(X[val_mask], y[val_mask])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
t_r2 = r2_score(y[val_mask], model_t.predict(X[val_mask]))
print(f"Time-based R²: {t_r2:.6f} ({t_r2*100:.2f}%)", flush=True)

# ── Main KFold baseline ──
print("\n--- KFold(5) baseline with new features ---", flush=True)
oof_v20, test_v20, r2_v20 = train_multiseed(X, y, X_test, BASE_SEEDS, BASE_PARAMS, cat_indices, tag='base')
print(f"\nBaseline v9 OOF was ~0.9590. New features OOF: {r2_v20:.6f}", flush=True)
PREV_BEST = 0.9590  # v9_bag reference
print(f"Delta vs v9: {(r2_v20 - PREV_BEST):+.4f}", flush=True)

# ── LGBM Feature Importance (instead of SHAP) ──
print("\n--- LGBM feature importance (top 20) ---", flush=True)
m_imp = lgb.LGBMRegressor(**BASE_PARAMS)
m_imp.fit(X, y, categorical_feature=cat_indices)
imp = pd.Series(m_imp.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(20).to_string(), flush=True)

# ── Optuna ──
print("\n--- Optuna 50 trials ---", flush=True)
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
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            subsample_freq=1,
            reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
            max_bin=trial.suggest_categorical('max_bin', [255, 511]),
            n_jobs=-1, verbose=-1, random_state=42
        )
        oof = np.zeros(len(X))
        for tr_idx, val_idx in kf_opt.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_indices,
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            oof[val_idx] = m.predict(X.iloc[val_idx])
        return r2_score(y, oof)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best_optuna_r2 = study.best_value
    best_params = study.best_params
    print(f"Best Optuna R²: {best_optuna_r2:.6f} ({best_optuna_r2*100:.2f}%)", flush=True)
    print(f"Best params: {best_params}", flush=True)

    opt_params = {**best_params, 'n_jobs': -1, 'verbose': -1}
    print("\nOptuna model multi-seed:", flush=True)
    oof_opt, test_opt, r2_opt = train_multiseed(X, y, X_test, BASE_SEEDS, opt_params, cat_indices, tag='optuna')

    sub_p2 = test[['Index']].copy()
    sub_p2['demand'] = np.clip(test_opt, 0, 1)
    sub_p2.to_csv(BASE_DIR / 'submission_phase2.csv', index=False)
    print(f"Saved submission_phase2.csv (mean={sub_p2['demand'].mean():.4f})", flush=True)

except Exception as e:
    print(f"Optuna failed: {e}", flush=True)
    r2_opt = r2_v20
    test_opt = test_v20.copy()
    best_params = {k: v for k, v in BASE_PARAMS.items()}

print("\n" + "="*60, flush=True)
print("PHASE 3: PUSH TO 93+", flush=True)
print("="*60, flush=True)

experiment_log = [
    {'name': 'v9_bag_baseline', 'oof_r2': PREV_BEST, 'delta': 0.0, 'notes': '91.5% LB reference'},
    {'name': 'v20_base_newfeatures', 'oof_r2': r2_v20, 'delta': r2_v20 - PREV_BEST, 'notes': 'new features + base params'},
    {'name': 'optuna_tuned', 'oof_r2': r2_opt, 'delta': r2_opt - r2_v20, 'notes': 'optuna 50 trials'},
]

best_r2 = max(r2_v20, r2_opt)
best_test = test_opt.copy() if r2_opt >= r2_v20 else test_v20.copy()

# ── Phase A: DART ──
print("\n[A] DART boosting...", flush=True)
DART_PARAMS = dict(
    boosting_type='dart', n_estimators=1000, learning_rate=0.03, num_leaves=127,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
    drop_rate=0.1, skip_drop=0.5, n_jobs=-1, verbose=-1
)
dart_oof_list, dart_test_list = [], []
for seed in BASE_SEEDS:
    p = {**DART_PARAMS, 'random_state': seed}
    oof_ = np.zeros(len(X))
    test_ = np.zeros(len(X_test))
    kf_ = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf_.split(X):
        m = lgb.LGBMRegressor(**p)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_indices)
        oof_[val_idx] = m.predict(X.iloc[val_idx])
        test_ += m.predict(X_test) / 5
    dart_oof_list.append(oof_)
    dart_test_list.append(test_)
    print(f"  DART seed={seed}: R²={r2_score(y, oof_):.6f}", flush=True)

dart_oof = np.mean(dart_oof_list, axis=0)
dart_test = np.mean(dart_test_list, axis=0)
r2_dart = r2_score(y, dart_oof)
delta_dart = r2_dart - best_r2
print(f"DART OOF R²: {r2_dart:.6f} ({r2_dart*100:.2f}%), delta vs best={delta_dart:+.4f}", flush=True)
experiment_log.append({'name': 'DART', 'oof_r2': r2_dart, 'delta': delta_dart, 'notes': 'drop=0.1 skip=0.5'})
if r2_dart > best_r2:
    best_r2 = r2_dart
    best_test = dart_test.copy()
    print("  -> NEW BEST!", flush=True)

# ── Phase B: XGBoost blend ──
print("\n[B] XGBoost blend...", flush=True)
try:
    import xgboost as xgb
    XGB_PARAMS = dict(
        n_estimators=1000, learning_rate=0.03, max_leaves=127,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        tree_method='hist', n_jobs=-1, verbosity=0, random_state=42
    )
    xgb_oof_list, xgb_test_list = [], []
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
        print(f"  XGB seed={seed}: R²={r2_score(y, oof_):.6f}", flush=True)

    xgb_oof = np.mean(xgb_oof_list, axis=0)
    xgb_test = np.mean(xgb_test_list, axis=0)
    r2_xgb = r2_score(y, xgb_oof)
    print(f"XGB OOF R²: {r2_xgb:.6f} ({r2_xgb*100:.2f}%)", flush=True)

    best_blend_r2, best_w = -np.inf, 1.0
    for w in np.arange(0.5, 0.95, 0.05):
        r2_b = r2_score(y, w * oof_v20 + (1-w) * xgb_oof)
        if r2_b > best_blend_r2:
            best_blend_r2, best_w = r2_b, w

    blend_oof = best_w * oof_v20 + (1-best_w) * xgb_oof
    blend_test = best_w * test_v20 + (1-best_w) * xgb_test
    delta_blend = best_blend_r2 - best_r2
    print(f"Best LGBM/XGB blend w={best_w:.2f}: R²={best_blend_r2:.6f} ({best_blend_r2*100:.2f}%), delta={delta_blend:+.4f}", flush=True)
    experiment_log.append({'name': f'XGB_blend_w{best_w:.2f}', 'oof_r2': best_blend_r2, 'delta': delta_blend, 'notes': f'lgbm*{best_w:.2f}+xgb*{1-best_w:.2f}'})
    if best_blend_r2 > best_r2:
        best_r2 = best_blend_r2
        best_test = blend_test.copy()
        print("  -> NEW BEST!", flush=True)
except Exception as e:
    print(f"XGB failed: {e}", flush=True)

# ── Phase C: Expanded seeds ──
print("\n[C] Expanded seeds [42,123,2024,0,7,999,2025,314]...", flush=True)
EXPANDED_SEEDS = [42, 123, 2024, 0, 7, 999, 2025, 314]
oof_exp_all, test_exp_all = [], []
for seed in EXPANDED_SEEDS:
    oof_ = np.zeros(len(X))
    test_ = np.zeros(len(X_test))
    kf_ = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf_.split(X):
        m = lgb.LGBMRegressor(**{**BASE_PARAMS, 'random_state': seed})
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_indices,
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_[val_idx] = m.predict(X.iloc[val_idx])
        test_ += m.predict(X_test) / 5
    oof_exp_all.append(oof_)
    test_exp_all.append(test_)
    print(f"  seed={seed}: R²={r2_score(y, oof_):.6f}, test_mean={test_.mean():.4f}", flush=True)

oof_exp = np.mean(oof_exp_all, axis=0)
test_exp = np.mean(test_exp_all, axis=0)
r2_exp = r2_score(y, oof_exp)
delta_exp = r2_exp - best_r2
print(f"8-seed expanded R²: {r2_exp:.6f} ({r2_exp*100:.2f}%), delta={delta_exp:+.4f}", flush=True)
print(f"test pred mean={test_exp.mean():.4f} (v9 target: 0.1302)", flush=True)
experiment_log.append({'name': 'expanded_8seeds', 'oof_r2': r2_exp, 'delta': delta_exp, 'notes': '8 seeds, check test mean'})
if r2_exp > best_r2:
    best_r2 = r2_exp
    best_test = test_exp.copy()
    print("  -> NEW BEST!", flush=True)

# ── Phase D: Residual model ──
print("\n[D] Residual model correction...", flush=True)
residuals = y.values - oof_v20
oof_resid = np.zeros(len(X))
kf_r = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in kf_r.split(X):
    m_r = lgb.LGBMRegressor(**{**BASE_PARAMS, 'n_estimators': 300, 'num_leaves': 63})
    m_r.fit(X.iloc[tr_idx], residuals[tr_idx], categorical_feature=cat_indices)
    oof_resid[val_idx] = m_r.predict(X.iloc[val_idx])
m_r_full = lgb.LGBMRegressor(**{**BASE_PARAMS, 'n_estimators': 300, 'num_leaves': 63})
m_r_full.fit(X, residuals, categorical_feature=cat_indices)
test_resid_full = m_r_full.predict(X_test)

best_resid_r2, best_alpha = -np.inf, 0.0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    combined_oof = oof_v20 + alpha * oof_resid
    r2_c = r2_score(y, combined_oof)
    print(f"  alpha={alpha}: R²={r2_c:.6f} ({r2_c*100:.2f}%)", flush=True)
    if r2_c > best_resid_r2:
        best_resid_r2, best_alpha = r2_c, alpha

delta_resid = best_resid_r2 - best_r2
print(f"Best residual (alpha={best_alpha}): R²={best_resid_r2:.6f}, delta={delta_resid:+.4f}", flush=True)
experiment_log.append({'name': f'residual_alpha{best_alpha}', 'oof_r2': best_resid_r2, 'delta': delta_resid, 'notes': f'v20+{best_alpha}×residual'})
if best_resid_r2 > best_r2:
    best_r2 = best_resid_r2
    best_test = np.clip(test_v20 + best_alpha * test_resid_full, 0, 1)
    print("  -> NEW BEST!", flush=True)

# ── Phase E: Optuna-tuned expanded seeds ──
print("\n[E] Optuna params + expanded seeds...", flush=True)
try:
    opt_params_clean = {k: v for k, v in best_params.items()}
    if 'n_jobs' not in opt_params_clean:
        opt_params_clean['n_jobs'] = -1
    opt_params_clean['verbose'] = -1

    oof_opt_exp_all, test_opt_exp_all = [], []
    for seed in BASE_SEEDS:
        oof_ = np.zeros(len(X))
        test_ = np.zeros(len(X_test))
        kf_ = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf_.split(X):
            m = lgb.LGBMRegressor(**{**opt_params_clean, 'random_state': seed})
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_indices,
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            oof_[val_idx] = m.predict(X.iloc[val_idx])
            test_ += m.predict(X_test) / 5
        oof_opt_exp_all.append(oof_)
        test_opt_exp_all.append(test_)
        print(f"  opt seed={seed}: R²={r2_score(y, oof_):.6f}", flush=True)

    oof_opt_exp = np.mean(oof_opt_exp_all, axis=0)
    test_opt_exp = np.mean(test_opt_exp_all, axis=0)
    r2_opt_exp = r2_score(y, oof_opt_exp)
    delta_opt_exp = r2_opt_exp - best_r2
    print(f"Optuna+3seeds R²: {r2_opt_exp:.6f} ({r2_opt_exp*100:.2f}%), delta={delta_opt_exp:+.4f}", flush=True)
    experiment_log.append({'name': 'optuna_3seeds', 'oof_r2': r2_opt_exp, 'delta': delta_opt_exp, 'notes': 'optuna params + 3 seeds'})
    if r2_opt_exp > best_r2:
        best_r2 = r2_opt_exp
        best_test = test_opt_exp.copy()
        print("  -> NEW BEST!", flush=True)
except Exception as e:
    print(f"Optuna expanded seeds failed: {e}", flush=True)

# ── Final: Train on ALL train data ──
print("\n--- FINAL model: train on all data ---", flush=True)
final_params = best_params if 'n_estimators' in best_params else BASE_PARAMS
final_params_clean = {**final_params, 'n_jobs': -1, 'verbose': -1}

final_test_preds = []
for seed in BASE_SEEDS:
    m_final = lgb.LGBMRegressor(**{**final_params_clean, 'random_state': seed})
    m_final.fit(X, y, categorical_feature=cat_indices)
    final_test_preds.append(m_final.predict(X_test))
    print(f"  final seed={seed}: pred_mean={m_final.predict(X_test).mean():.4f}", flush=True)

final_preds = np.clip(np.mean(final_test_preds, axis=0), 0, 1)
best_test_clipped = np.clip(best_test, 0, 1)

# ── Save submissions ──
sub_v20 = test[['Index']].copy()
sub_v20['demand'] = best_test_clipped
sub_v20.to_csv(BASE_DIR / 'submission_v20_newfeatures.csv', index=False)
print(f"Saved submission_v20_newfeatures.csv ({len(sub_v20)} rows)", flush=True)

sub_alltrain = test[['Index']].copy()
sub_alltrain['demand'] = final_preds
sub_alltrain.to_csv(BASE_DIR / 'submission_v20_alltrain.csv', index=False)
print(f"Saved submission_v20_alltrain.csv", flush=True)

print("\n" + "="*60, flush=True)
print("EXPERIMENT SUMMARY TABLE", flush=True)
print("="*60, flush=True)
print(f"{'Name':<35} {'OOF R²':>10} {'Score%':>8} {'Delta':>8}  Notes", flush=True)
print("-"*80, flush=True)
for exp in experiment_log:
    score = exp['oof_r2'] * 100
    print(f"{exp['name']:<35} {exp['oof_r2']:>10.6f} {score:>7.2f}% {exp['delta']:>+8.4f}  {exp.get('notes','')}", flush=True)

print("\n" + "="*60, flush=True)
print("FINAL RESULTS", flush=True)
print("="*60, flush=True)
print(f"Best OOF R²:  {best_r2:.6f} → {best_r2*100:.2f}%", flush=True)
print(f"Previous best (v9): 0.9590 → 91.5% LB", flush=True)
print(f"OOF improvement: {(best_r2 - PREV_BEST):+.4f} ({(best_r2-PREV_BEST)*100:+.2f}pp)", flush=True)
print(f"Estimated LB: {(best_r2-0.045)*100:.2f}%  (OOF-to-LB gap ≈4.5pp)", flush=True)
print(f"\nSubmission stats:", flush=True)
print(f"  mean={best_test_clipped.mean():.4f}, std={best_test_clipped.std():.4f}", flush=True)
print(f"  min={best_test_clipped.min():.4f}, max={best_test_clipped.max():.4f}", flush=True)
print(f"\nFormat check: {sub_v20.shape[0]}×{sub_v20.shape[1]}, cols={list(sub_v20.columns)}", flush=True)
print("DONE", flush=True)
