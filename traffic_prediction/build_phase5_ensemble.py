"""Replace cells 57-76 (old Phases 5-9) with the comprehensive ensemble + stacking pipeline."""
import json, uuid

NB = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/notebooks/flipkart_traffic_prediction.ipynb"

def nid(): return uuid.uuid4().hex[:8]
def code(src): return {"cell_type":"code","execution_count":None,"id":nid(),"metadata":{},"outputs":[],"source":src if isinstance(src,list) else [src]}
def md(src):   return {"cell_type":"markdown","id":nid(),"metadata":{},"source":src if isinstance(src,list) else [src]}

cells = []

# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Modeling setup
# ════════════════════════════════════════════════════════════════════════════
cells.append(md([
    "---\n",
    "## Phase 5 — Modeling Setup & Validation Strategy\n",
    "\n",
    "Loads the engineered features from `/features/`, defines the metric and CV.\n",
    "\n",
    "**Validation:** aligned `KFold(5, shuffle=True, random_state=42)` — *identical* to the Phase 4 target-encoding folds, so OOF target encoding and model CV use the same splits (leak-free). Test is ~99% seen geohashes, so this mirrors the leaderboard.\n",
    "\n",
    "**Metric:** `R²` (sklearn). **Competition score:** `max(0, 100 · R²)`."
]))
cells.append(code([
    "# 5.0 — Load engineered features + set up CV / metric\n",
    "from sklearn.model_selection import KFold, cross_val_predict\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.metrics import r2_score, mean_squared_error\n",
    "from scipy.optimize import minimize\n",
    "import optuna, joblib, time\n",
    "from optuna.samplers import TPESampler\n",
    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
    "\n",
    "train_feat = pd.read_csv(f'{OUT}/features/train_features.csv')\n",
    "test_feat  = pd.read_csv(f'{OUT}/features/test_features.csv')\n",
    "\n",
    "FEATURES   = [c for c in train_feat.columns if c not in ('Index', 'demand')]\n",
    "X          = train_feat[FEATURES].copy()\n",
    "y          = train_feat['demand'].values\n",
    "X_test     = test_feat[FEATURES].copy()\n",
    "test_index = test_feat['Index'].values\n",
    "\n",
    "# Categorical columns (integer-coded labels, not magnitudes) for CatBoost\n",
    "CAT_FEATURES = [c for c in ['geo_cluster','geohash5_enc','geohash4_enc',\n",
    "                            'time_of_day','day_of_week','temp_binned'] if c in FEATURES]\n",
    "for c in CAT_FEATURES:\n",
    "    X[c]      = X[c].astype(int)\n",
    "    X_test[c] = X_test[c].astype(int)\n",
    "\n",
    "N_FOLDS = 5\n",
    "kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)   # == Phase 4 fe_kf\n",
    "\n",
    "def comp_score(actual, pred):\n",
    "    return max(0.0, 100.0 * r2_score(actual, pred))\n",
    "def rmse(a, p):\n",
    "    return np.sqrt(mean_squared_error(a, p))\n",
    "\n",
    "# Fast single-holdout split for Optuna tuning (fold-0 of the aligned CV)\n",
    "_tr_i, _va_i = next(kf.split(X))\n",
    "Xtt, Xtv = X.iloc[_tr_i], X.iloc[_va_i]\n",
    "ytt, ytv = y[_tr_i], y[_va_i]\n",
    "\n",
    "def make_cb(name, every=25):\n",
    "    def _cb(study, trial):\n",
    "        if (trial.number + 1) % every == 0:\n",
    "            print(f'    [{name}] trial {trial.number+1:3d}  best R²={study.best_value:.5f}')\n",
    "    return _cb\n",
    "\n",
    "print(f'X={X.shape}  X_test={X_test.shape}  features={len(FEATURES)}')\n",
    "print(f'CatBoost cat_features: {CAT_FEATURES}')\n",
    "\n",
    "# generic 5-fold OOF runner\n",
    "def run_cv(make_model, fit_one, name, save=True):\n",
    "    oof  = np.zeros(len(X)); test_pred = np.zeros(len(X_test)); scores = []; models = []\n",
    "    for fold, (ti, vi) in enumerate(kf.split(X), 1):\n",
    "        m = make_model()\n",
    "        fit_one(m, X.iloc[ti], y[ti], X.iloc[vi], y[vi])\n",
    "        oof[vi]    = m.predict(X.iloc[vi])\n",
    "        test_pred += m.predict(X_test) / N_FOLDS\n",
    "        s = r2_score(y[vi], oof[vi]); scores.append(s); models.append(m)\n",
    "        print(f'    [{name}] fold {fold}: R²={s:.5f}  RMSE={rmse(y[vi], oof[vi]):.5f}')\n",
    "    print(f'    [{name}] OOF R²={r2_score(y, oof):.5f} | per-fold {np.mean(scores):.5f} ± {np.std(scores):.5f}')\n",
    "    if save:\n",
    "        joblib.dump(models, f'{OUT}/models/{name}_fold_models.pkl')\n",
    "    return oof, test_pred, scores, models"
]))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 6 — LightGBM
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(["---\n", "## Phase 6 — Model 1: LightGBM (Optuna 150 trials)"]))
cells.append(code([
    "# 6.1 — LightGBM Optuna tuning (single holdout, early stopping)\n",
    "def lgb_objective(trial):\n",
    "    p = dict(objective='regression', metric='rmse',\n",
    "             num_leaves        = trial.suggest_int('num_leaves', 31, 512),\n",
    "             learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),\n",
    "             feature_fraction  = trial.suggest_float('feature_fraction', 0.5, 1.0),\n",
    "             min_child_samples = trial.suggest_int('min_child_samples', 5, 100),\n",
    "             bagging_fraction=0.8, bagging_freq=5, n_estimators=1500,\n",
    "             random_state=SEED, n_jobs=-1, verbose=-1)\n",
    "    m = lgb.LGBMRegressor(**p)\n",
    "    m.fit(Xtt, ytt, eval_set=[(Xtv, ytv)],\n",
    "          callbacks=[lgb.early_stopping(100, verbose=False)])\n",
    "    return r2_score(ytv, m.predict(Xtv))\n",
    "\n",
    "t0 = time.time()\n",
    "study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))\n",
    "study_lgb.optimize(lgb_objective, n_trials=150, callbacks=[make_cb('LGB')], show_progress_bar=False)\n",
    "print(f'\\nLGB best holdout R²={study_lgb.best_value:.5f}  ({time.time()-t0:.0f}s)')\n",
    "print('LGB best params:', study_lgb.best_params)\n",
    "\n",
    "best_lgb = dict(study_lgb.best_params)\n",
    "best_lgb.update(objective='regression', metric='rmse', bagging_fraction=0.8,\n",
    "                bagging_freq=5, n_estimators=3000, random_state=SEED, n_jobs=-1, verbose=-1)\n",
    "json.dump(best_lgb, open(f'{OUT}/models/best_lgb_params.json', 'w'), indent=2)"
]))
cells.append(code([
    "# 6.2 — LightGBM 5-fold OOF\n",
    "def fit_lgb(m, Xtr, ytr, Xva, yva):\n",
    "    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])\n",
    "\n",
    "print('LightGBM 5-fold:')\n",
    "oof_lgb, test_lgb, sc_lgb, models_lgb = run_cv(\n",
    "    lambda: lgb.LGBMRegressor(**best_lgb), fit_lgb, 'lgb')"
]))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 7 — XGBoost
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(["---\n", "## Phase 7 — Model 2: XGBoost (Optuna 100 trials)"]))
cells.append(code([
    "# 7.1 — XGBoost Optuna tuning\n",
    "def xgb_objective(trial):\n",
    "    p = dict(n_estimators=1500,\n",
    "             learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),\n",
    "             max_depth        = trial.suggest_int('max_depth', 3, 12),\n",
    "             subsample        = trial.suggest_float('subsample', 0.5, 1.0),\n",
    "             colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),\n",
    "             tree_method='hist', random_state=SEED, n_jobs=-1,\n",
    "             early_stopping_rounds=100, eval_metric='rmse')\n",
    "    m = xgb.XGBRegressor(**p)\n",
    "    m.fit(Xtt, ytt, eval_set=[(Xtv, ytv)], verbose=False)\n",
    "    return r2_score(ytv, m.predict(Xtv))\n",
    "\n",
    "t0 = time.time()\n",
    "study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))\n",
    "study_xgb.optimize(xgb_objective, n_trials=100, callbacks=[make_cb('XGB')], show_progress_bar=False)\n",
    "print(f'\\nXGB best holdout R²={study_xgb.best_value:.5f}  ({time.time()-t0:.0f}s)')\n",
    "print('XGB best params:', study_xgb.best_params)\n",
    "\n",
    "best_xgb = dict(study_xgb.best_params)\n",
    "best_xgb.update(n_estimators=3000, tree_method='hist', random_state=SEED, n_jobs=-1,\n",
    "                early_stopping_rounds=100, eval_metric='rmse')\n",
    "json.dump(best_xgb, open(f'{OUT}/models/best_xgb_params.json', 'w'), indent=2)"
]))
cells.append(code([
    "# 7.2 — XGBoost 5-fold OOF\n",
    "def fit_xgb(m, Xtr, ytr, Xva, yva):\n",
    "    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)\n",
    "\n",
    "print('XGBoost 5-fold:')\n",
    "oof_xgb, test_xgb, sc_xgb, models_xgb = run_cv(\n",
    "    lambda: xgb.XGBRegressor(**best_xgb), fit_xgb, 'xgb')"
]))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 8 — CatBoost
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(["---\n", "## Phase 8 — Model 3: CatBoost (Optuna 100 trials)"]))
cells.append(code([
    "# 8.1 — CatBoost Optuna tuning (with cat_features)\n",
    "def cat_objective(trial):\n",
    "    p = dict(iterations=1000, learning_rate=0.05,\n",
    "             depth               = trial.suggest_int('depth', 4, 10),\n",
    "             l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 1, 10),\n",
    "             bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1),\n",
    "             random_seed=SEED, eval_metric='RMSE', od_type='Iter', od_wait=100, verbose=False)\n",
    "    m = CatBoostRegressor(**p)\n",
    "    m.fit(Xtt, ytt, eval_set=(Xtv, ytv), cat_features=CAT_FEATURES,\n",
    "          use_best_model=True, verbose=False)\n",
    "    return r2_score(ytv, m.predict(Xtv))\n",
    "\n",
    "t0 = time.time()\n",
    "study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))\n",
    "study_cat.optimize(cat_objective, n_trials=100, callbacks=[make_cb('CAT')], show_progress_bar=False)\n",
    "print(f'\\nCAT best holdout R²={study_cat.best_value:.5f}  ({time.time()-t0:.0f}s)')\n",
    "print('CAT best params:', study_cat.best_params)\n",
    "\n",
    "best_cat = dict(study_cat.best_params)\n",
    "best_cat.update(iterations=3000, learning_rate=0.05, random_seed=SEED,\n",
    "                eval_metric='RMSE', od_type='Iter', od_wait=100, verbose=False)\n",
    "json.dump(best_cat, open(f'{OUT}/models/best_cat_params.json', 'w'), indent=2)"
]))
cells.append(code([
    "# 8.2 — CatBoost 5-fold OOF\n",
    "def fit_cat(m, Xtr, ytr, Xva, yva):\n",
    "    m.fit(Xtr, ytr, eval_set=(Xva, yva), cat_features=CAT_FEATURES,\n",
    "          use_best_model=True, verbose=False)\n",
    "\n",
    "print('CatBoost 5-fold:')\n",
    "oof_cat, test_cat, sc_cat, models_cat = run_cv(\n",
    "    lambda: CatBoostRegressor(**best_cat), fit_cat, 'cat')"
]))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 9 — Stacking
# ════════════════════════════════════════════════════════════════════════════
cells.append(md([
    "---\n",
    "## Phase 9 — Stacking Ensemble\n",
    "Ridge meta-learner on OOF predictions (honest meta-CV via `cross_val_predict`) **and** a scipy-optimized weighted blend. The higher-OOF-R² option is used for the final test prediction."
]))
cells.append(code([
    "# 9.1 — Stack OOF predictions; Ridge meta-learner + weighted blend\n",
    "S_oof  = np.column_stack([oof_lgb, oof_xgb, oof_cat])\n",
    "S_test = np.column_stack([test_lgb, test_xgb, test_cat])\n",
    "\n",
    "# (a) Ridge meta-learner — honest OOF via cross_val_predict on the meta-features\n",
    "ridge = Ridge(alpha=1.0)\n",
    "meta_oof  = cross_val_predict(ridge, S_oof, y, cv=kf)\n",
    "ridge.fit(S_oof, y)\n",
    "meta_test = ridge.predict(S_test)\n",
    "r2_meta   = r2_score(y, meta_oof)\n",
    "print(f'Ridge meta  : coef={ridge.coef_.round(4)} intercept={ridge.intercept_:.5f}')\n",
    "print(f'Ridge stack OOF R² = {r2_meta:.5f}')\n",
    "\n",
    "# (b) Weighted blend — maximize R² over non-negative weights summing to 1\n",
    "def neg_r2(w):\n",
    "    w = np.clip(w, 0, None); s = w.sum()\n",
    "    w = w / s if s > 0 else w\n",
    "    return -r2_score(y, S_oof @ w)\n",
    "res = minimize(neg_r2, [1/3, 1/3, 1/3], method='Nelder-Mead',\n",
    "               options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 3000})\n",
    "w = np.clip(res.x, 0, None); w = w / w.sum()\n",
    "blend_oof  = S_oof  @ w\n",
    "blend_test = S_test @ w\n",
    "r2_blend   = r2_score(y, blend_oof)\n",
    "print(f'Blend weights: LGB={w[0]:.3f} XGB={w[1]:.3f} CAT={w[2]:.3f}')\n",
    "print(f'Weighted blend OOF R² = {r2_blend:.5f}')\n",
    "\n",
    "# (c) choose the better strategy for the final prediction\n",
    "if r2_meta >= r2_blend:\n",
    "    final_oof, final_test, chosen = meta_oof, meta_test, 'Ridge stack'\n",
    "else:\n",
    "    final_oof, final_test, chosen = blend_oof, blend_test, f'Blend (LGB={w[0]:.2f},XGB={w[1]:.2f},CAT={w[2]:.2f})'\n",
    "final_test = np.clip(final_test, 0, 1)\n",
    "\n",
    "print(f'\\n>>> Chosen ensemble : {chosen}')\n",
    "print(f'>>> Final ensemble OOF R²       = {r2_score(y, final_oof):.5f}')\n",
    "print(f'>>> Final competition score      = {comp_score(y, final_oof):.4f}  (max 0, 100·R²)')\n",
    "\n",
    "# save stacking artifacts\n",
    "joblib.dump({'ridge': ridge, 'weights': w, 'chosen': chosen,\n",
    "             'r2_meta': r2_meta, 'r2_blend': r2_blend},\n",
    "            f'{OUT}/models/stacking.pkl')\n",
    "np.save(f'{OUT}/models/oof_preds.npy', S_oof)\n",
    "np.save(f'{OUT}/models/test_preds.npy', S_test)\n",
    "print('Saved stacking.pkl, oof_preds.npy, test_preds.npy')"
]))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 10 — SHAP + comparison
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(["---\n", "## Phase 10 — SHAP Feature Importance (per model) & Comparison"]))
cells.append(code([
    "# 10.1 — Top-20 SHAP importance per model (fold-0 models; sampled rows)\n",
    "import shap\n",
    "from catboost import Pool\n",
    "\n",
    "samp = X.sample(min(3000, len(X)), random_state=SEED)\n",
    "shap_tables = {}\n",
    "\n",
    "def top_shap(name, model, kind):\n",
    "    try:\n",
    "        if kind == 'cat':\n",
    "            pool = Pool(samp, cat_features=CAT_FEATURES)\n",
    "            sv = model.get_feature_importance(pool, type='ShapValues')[:, :-1]\n",
    "        else:\n",
    "            sv = shap.TreeExplainer(model).shap_values(samp)\n",
    "        imp = np.abs(sv).mean(axis=0)\n",
    "        src = 'SHAP'\n",
    "    except Exception as e:\n",
    "        print(f'  [{name}] SHAP failed ({type(e).__name__}); using native importance')\n",
    "        imp = model.feature_importances_; src = 'native'\n",
    "    tbl = (pd.DataFrame({'feature': FEATURES, 'importance': imp})\n",
    "             .sort_values('importance', ascending=False).reset_index(drop=True))\n",
    "    shap_tables[name] = (tbl, src)\n",
    "    return tbl\n",
    "\n",
    "tbl_lgb = top_shap('LightGBM', models_lgb[0], 'lgb')\n",
    "tbl_xgb = top_shap('XGBoost',  models_xgb[0], 'xgb')\n",
    "tbl_cat = top_shap('CatBoost', models_cat[0], 'cat')\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(22, 8))\n",
    "for ax, (name, (tbl, src)) in zip(axes, shap_tables.items()):\n",
    "    sns.barplot(data=tbl.head(20), x='importance', y='feature', palette='viridis', ax=ax)\n",
    "    ax.set_title(f'{name} — Top 20 ({src})')\n",
    "    ax.set_xlabel(f'mean(|{src}|)')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{OUT}/features/shap_per_model.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "for name, (tbl, src) in shap_tables.items():\n",
    "    print(f'\\n=== {name} top 20 ({src}) ===')\n",
    "    print(tbl.head(20).to_string(index=False))"
]))
cells.append(code([
    "# 10.2 — Model comparison (CV R² mean ± std)\n",
    "rows = [\n",
    "    ('LightGBM',       r2_score(y, oof_lgb), np.mean(sc_lgb), np.std(sc_lgb)),\n",
    "    ('XGBoost',        r2_score(y, oof_xgb), np.mean(sc_xgb), np.std(sc_xgb)),\n",
    "    ('CatBoost',       r2_score(y, oof_cat), np.mean(sc_cat), np.std(sc_cat)),\n",
    "    ('Ridge stack',    r2_meta, np.nan, np.nan),\n",
    "    ('Weighted blend', r2_blend, np.nan, np.nan),\n",
    "]\n",
    "comp = pd.DataFrame(rows, columns=['model', 'oof_r2', 'fold_mean', 'fold_std'])\n",
    "comp['comp_score'] = (100 * comp['oof_r2']).clip(lower=0)\n",
    "print(comp.to_string(index=False))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "colors = ['#4CAF50' if v == comp['oof_r2'].max() else '#2196F3' for v in comp['oof_r2']]\n",
    "ax.barh(comp['model'], comp['oof_r2'], color=colors, edgecolor='white')\n",
    "ax.set_xlim(min(comp['oof_r2']) - 0.005, 1.0)\n",
    "ax.axvline(0.95, color='red', linestyle='--', label='target R²=0.95')\n",
    "for i, v in enumerate(comp['oof_r2']):\n",
    "    ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=9)\n",
    "ax.set_title('OOF R² by model'); ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{OUT}/models/model_comparison.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ════════════════════════════════════════════════════════════════════════════
# PHASE 11 — Submission
# ════════════════════════════════════════════════════════════════════════════
cells.append(md(["---\n", "## Phase 11 — Generate Submission"]))
cells.append(code([
    "# 11.1 — Build submission from the chosen ensemble\n",
    "submission = pd.DataFrame({'Index': test_index, 'demand': final_test})\n",
    "sub_path = f'{OUT}/submissions/submission_stack.csv'\n",
    "submission.to_csv(sub_path, index=False)\n",
    "\n",
    "print(f'Saved -> {sub_path}  shape={submission.shape}')\n",
    "print(f'pred: min={final_test.min():.5f} max={final_test.max():.5f} '\n",
    "      f'mean={final_test.mean():.5f} median={np.median(final_test):.5f}')\n",
    "submission.head(10)"
]))
cells.append(code([
    "# 11.2 — Prediction distribution + final summary\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
    "axes[0].hist(y, bins=80, alpha=0.6, label='train actual', color='steelblue')\n",
    "axes[0].hist(final_test, bins=80, alpha=0.6, label='test pred', color='coral')\n",
    "axes[0].set_title('Demand: train vs predicted'); axes[0].legend()\n",
    "axes[1].hist(np.log1p(y), bins=80, alpha=0.6, label='train actual', color='steelblue')\n",
    "axes[1].hist(np.log1p(final_test), bins=80, alpha=0.6, label='test pred', color='coral')\n",
    "axes[1].set_title('log1p(demand): train vs predicted'); axes[1].legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{OUT}/submissions/prediction_distribution.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print('=' * 64)\n",
    "print('FINAL RESULTS')\n",
    "print('=' * 64)\n",
    "print(f'  LightGBM       OOF R² = {r2_score(y, oof_lgb):.5f}')\n",
    "print(f'  XGBoost        OOF R² = {r2_score(y, oof_xgb):.5f}')\n",
    "print(f'  CatBoost       OOF R² = {r2_score(y, oof_cat):.5f}')\n",
    "print(f'  Ridge stack    OOF R² = {r2_meta:.5f}')\n",
    "print(f'  Weighted blend OOF R² = {r2_blend:.5f}')\n",
    "print('-' * 64)\n",
    "print(f'  CHOSEN         : {chosen}')\n",
    "print(f'  Final OOF R²   : {r2_score(y, final_oof):.5f}')\n",
    "print(f'  Competition    : {comp_score(y, final_oof):.4f}  (target > 95)')\n",
    "print('=' * 64)\n",
    "print('Models saved to /models/:')\n",
    "import os as _os\n",
    "for f in sorted(_os.listdir(f'{OUT}/models')):\n",
    "    print('   ', f)"
]))

# ─────────────────────────────────────────────────────────────────────────────
with open(NB) as f:
    nb = json.load(f)

lo, hi = 57, 76   # replace old Phases 5-9
nb['cells'] = nb['cells'][:lo] + cells + nb['cells'][hi+1:]

seen = set()
for c in nb['cells']:
    v = uuid.uuid4().hex[:8]
    while v in seen: v = uuid.uuid4().hex[:8]
    c['id'] = v; seen.add(v)
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Replaced cells {lo}-{hi} with {len(cells)} new cells. Total now: {len(nb['cells'])}")
