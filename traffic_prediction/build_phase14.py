"""Append Phase 14 — Final Submission to the notebook."""
import json, uuid

NB = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/notebooks/flipkart_traffic_prediction.ipynb"

def nid(): return uuid.uuid4().hex[:8]
def code(src): return {"cell_type":"code","execution_count":None,"id":nid(),"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","id":nid(),"metadata":{},"source":src}

cells = []

# ── Phase 14 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Phase 14 — Final Submission\n"
    "\n"
    "Retrains all three models on the **entire training set** (n_estimators=3000) using the best "
    "hyperparameters from Optuna, then blends with the Ridge stacking weights learned during CV. "
    "This is standard practice: CV finds the right hyperparameters and blend weights, "
    "full-data training gives the strongest model (no rows held out).\n"
    "\n"
    "| Step | Detail |\n"
    "|------|---------|\n"
    "| Features | V1 features from `/features/train_features.csv` (50 features) |\n"
    "| Models | LGB + XGB + CatBoost, each with n_estimators=3000 on 100% of train |\n"
    "| Blend | Ridge stacking weights from Phase 9 CV |\n"
    "| Post-process | clip to [0, 1] — demand is naturally bounded |\n"
    "| Output | `submissions/final_submission.csv` (41 778 × 2) |"
))

# ── 14.0 Load & validate features ────────────────────────────────────────
cells.append(code(
    "# 14.0 — Load features + best CV artefacts\n"
    "print('Loading engineered features...')\n"
    "train_feat = pd.read_csv(f'{OUT}/features/train_features.csv')\n"
    "test_feat  = pd.read_csv(f'{OUT}/features/test_features.csv')\n"
    "sub_tmpl   = pd.read_csv(f'{BASE}/sample_submission.csv')\n"
    "\n"
    "FEAT_COLS  = [c for c in train_feat.columns if c not in ('Index','demand')]\n"
    "Xfull      = train_feat[FEAT_COLS].copy()\n"
    "yfull      = train_feat['demand'].values\n"
    "Xtest_full = test_feat[FEAT_COLS].copy()\n"
    "test_idx   = test_feat['Index'].values\n"
    "\n"
    "# CatBoost integer-typed columns\n"
    "CAT_COLS = [c for c in ['geo_cluster','geohash5_enc','geohash4_enc',\n"
    "                        'time_of_day','day_of_week','temp_binned'] if c in FEAT_COLS]\n"
    "for c in CAT_COLS:\n"
    "    Xfull[c]      = Xfull[c].astype(int)\n"
    "    Xtest_full[c] = Xtest_full[c].astype(int)\n"
    "\n"
    "# Residual null safety\n"
    "Xfull      = Xfull.fillna(Xfull.median())\n"
    "Xtest_full = Xtest_full.fillna(Xfull.median())\n"
    "\n"
    "# Load best CV stacking weights (Ridge trained on 5-fold OOF)\n"
    "stk_v1 = joblib.load(f'{OUT}/models/stacking.pkl')\n"
    "ridge_w = stk_v1['ridge'].coef_          # [lgb_w, xgb_w, cat_w]\n"
    "ridge_i = stk_v1['ridge'].intercept_\n"
    "oof_r2  = stk_v1['r2_meta']              # CV R² from Phase 9\n"
    "\n"
    "print(f'Train features : {Xfull.shape}')\n"
    "print(f'Test  features : {Xtest_full.shape}')\n"
    "print(f'Feature count  : {len(FEAT_COLS)}')\n"
    "print(f'CatBoost cats  : {CAT_COLS}')\n"
    "print(f'Nulls train/test: {Xfull.isnull().sum().sum()} / {Xtest_full.isnull().sum().sum()}')\n"
    "print(f'CV Ridge weights: LGB={ridge_w[0]:.4f}  XGB={ridge_w[1]:.4f}  CAT={ridge_w[2]:.4f}  intercept={ridge_i:.5f}')\n"
    "print(f'Best CV R²      : {oof_r2:.6f}  →  competition score: {100*oof_r2:.4f}')"
))

# ── 14.1 Load / fix best params ───────────────────────────────────────────
cells.append(code(
    "# 14.1 — Load best hyperparameters; fix n_estimators for full-data training\n"
    "#         (smoke-test params have n_estimators=120; override to 3000)\n"
    "import copy\n"
    "\n"
    "def fix_params(raw, model):\n"
    "    p = copy.deepcopy(raw)\n"
    "    if model == 'lgb':\n"
    "        p['n_estimators'] = 3000\n"
    "        p.update(verbose=-1, n_jobs=-1, random_state=SEED)\n"
    "    elif model == 'xgb':\n"
    "        p['n_estimators'] = 3000\n"
    "        # remove early_stopping_rounds — no eval_set on full data\n"
    "        p.pop('early_stopping_rounds', None)\n"
    "        p.update(tree_method='hist', random_state=SEED, n_jobs=-1)\n"
    "    elif model == 'cat':\n"
    "        p['iterations'] = 3000\n"
    "        p.update(random_seed=SEED, eval_metric='RMSE', verbose=False)\n"
    "        p.pop('od_type', None); p.pop('od_wait', None)  # no early stop on full data\n"
    "    return p\n"
    "\n"
    "p_lgb = fix_params(json.load(open(f'{OUT}/models/best_lgb_params.json')), 'lgb')\n"
    "p_xgb = fix_params(json.load(open(f'{OUT}/models/best_xgb_params.json')), 'xgb')\n"
    "p_cat = fix_params(json.load(open(f'{OUT}/models/best_cat_params.json')), 'cat')\n"
    "\n"
    "print('LGB params :', {k:v for k,v in p_lgb.items() if k not in ('n_jobs','verbose','random_state')})\n"
    "print('XGB params :', {k:v for k,v in p_xgb.items() if k not in ('n_jobs','tree_method','random_state')})\n"
    "print('CAT params :', {k:v for k,v in p_cat.items() if k not in ('random_seed','eval_metric','verbose')})"
))

# ── 14.2 Train full-data models ───────────────────────────────────────────
cells.append(code(
    "# 14.2 — Train LightGBM on full training data\n"
    "print('Training LightGBM (3000 trees, full data)...')\n"
    "t0 = time.time()\n"
    "final_lgb = lgb.LGBMRegressor(**p_lgb)\n"
    "final_lgb.fit(Xfull, yfull)\n"
    "pred_lgb = final_lgb.predict(Xtest_full)\n"
    "print(f'  done in {time.time()-t0:.0f}s  pred: min={pred_lgb.min():.4f}  max={pred_lgb.max():.4f}  mean={pred_lgb.mean():.4f}')\n"
    "joblib.dump(final_lgb, f'{OUT}/models/final_lgb.pkl')\n"
    "print('  saved -> models/final_lgb.pkl')"
))

cells.append(code(
    "# 14.3 — Train XGBoost on full training data\n"
    "print('Training XGBoost (3000 trees, full data)...')\n"
    "t0 = time.time()\n"
    "final_xgb = xgb.XGBRegressor(**p_xgb)\n"
    "final_xgb.fit(Xfull, yfull, verbose=False)\n"
    "pred_xgb = final_xgb.predict(Xtest_full)\n"
    "print(f'  done in {time.time()-t0:.0f}s  pred: min={pred_xgb.min():.4f}  max={pred_xgb.max():.4f}  mean={pred_xgb.mean():.4f}')\n"
    "joblib.dump(final_xgb, f'{OUT}/models/final_xgb.pkl')\n"
    "print('  saved -> models/final_xgb.pkl')"
))

cells.append(code(
    "# 14.4 — Train CatBoost on full training data\n"
    "print('Training CatBoost (3000 iterations, full data)...')\n"
    "t0 = time.time()\n"
    "final_cat = CatBoostRegressor(**p_cat)\n"
    "final_cat.fit(Xfull, yfull, cat_features=CAT_COLS)\n"
    "pred_cat = final_cat.predict(Xtest_full)\n"
    "print(f'  done in {time.time()-t0:.0f}s  pred: min={pred_cat.min():.4f}  max={pred_cat.max():.4f}  mean={pred_cat.mean():.4f}')\n"
    "joblib.dump(final_cat, f'{OUT}/models/final_cat.pkl')\n"
    "print('  saved -> models/final_cat.pkl')"
))

# ── 14.5 Blend + post-process ─────────────────────────────────────────────
cells.append(code(
    "# 14.5 — Ridge blend + post-processing\n"
    "# Stack predictions and apply CV-derived Ridge weights\n"
    "S_final = np.column_stack([pred_lgb, pred_xgb, pred_cat])\n"
    "pred_blend = S_final @ ridge_w + ridge_i           # Ridge linear combination\n"
    "\n"
    "# No log1p was applied — demand was predicted in original space\n"
    "# Clip to valid demand range [0, 1]\n"
    "pred_final = np.clip(pred_blend, 0.0, 1.0)\n"
    "\n"
    "print(f'Raw blend stats  : min={pred_blend.min():.6f}  max={pred_blend.max():.6f}  '\n"
    "      f'mean={pred_blend.mean():.6f}  std={pred_blend.std():.6f}')\n"
    "print(f'Clipped <0       : {(pred_blend < 0).sum():5d}  ({100*(pred_blend<0).mean():.2f}%)')\n"
    "print(f'Clipped >1       : {(pred_blend > 1).sum():5d}  ({100*(pred_blend>1).mean():.2f}%)')\n"
    "print(f'Final pred stats : min={pred_final.min():.6f}  max={pred_final.max():.6f}  '\n"
    "      f'mean={pred_final.mean():.6f}  std={pred_final.std():.6f}')"
))

# ── 14.6 Sanity checks ────────────────────────────────────────────────────
cells.append(code(
    "# 14.6 — Full sanity checks\n"
    "print('=' * 60)\n"
    "print('SANITY CHECKS')\n"
    "print('=' * 60)\n"
    "\n"
    "checks = []\n"
    "def chk(name, cond, detail=''):\n"
    "    status = '✓ PASS' if cond else '✗ FAIL'\n"
    "    checks.append(cond)\n"
    "    print(f'  {status}  {name}  {detail}')\n"
    "\n"
    "chk('Count = 41778',      len(pred_final) == 41778,         f'got {len(pred_final)}')\n"
    "chk('No nulls',           ~np.isnan(pred_final).any(),      f'nulls: {np.isnan(pred_final).sum()}')\n"
    "chk('Min >= 0',           pred_final.min() >= 0,            f'min={pred_final.min():.6f}')\n"
    "chk('Max <= 1',           pred_final.max() <= 1.0,          f'max={pred_final.max():.6f}')\n"
    "chk('No infs',            ~np.isinf(pred_final).any(),      f'infs: {np.isinf(pred_final).sum()}')\n"
    "\n"
    "# distribution similarity\n"
    "train_mean, train_std = yfull.mean(), yfull.std()\n"
    "pred_mean,  pred_std  = pred_final.mean(), pred_final.std()\n"
    "mean_ok = abs(pred_mean - train_mean) < 0.02\n"
    "std_ok  = abs(pred_std  - train_std)  < 0.04\n"
    "chk('Mean similar to train', mean_ok, f'pred={pred_mean:.4f}  train={train_mean:.4f}  diff={abs(pred_mean-train_mean):.4f}')\n"
    "chk('Std  similar to train', std_ok,  f'pred={pred_std:.4f}   train={train_std:.4f}   diff={abs(pred_std-train_std):.4f}')\n"
    "\n"
    "# index alignment\n"
    "chk('Index matches sample_sub', list(test_idx[:5]) == list(sub_tmpl['Index'][:5].values), f'first 5: {test_idx[:5].tolist()}')\n"
    "\n"
    "print('=' * 60)\n"
    "print(f'  Passed: {sum(checks)}/{len(checks)}')\n"
    "if not all(checks): print('  *** REVIEW FAILED CHECKS BEFORE SUBMITTING ***')\n"
    "\n"
    "# Distribution comparison plot\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "axes[0].hist(yfull,      bins=80, alpha=0.6, color='steelblue', label=f'Train actual (μ={train_mean:.3f})')\n"
    "axes[0].hist(pred_final, bins=80, alpha=0.6, color='coral',     label=f'Test pred   (μ={pred_mean:.3f})')\n"
    "axes[0].set_title('Demand Distribution: Train vs Final Predictions')\n"
    "axes[0].legend()\n"
    "axes[1].hist(np.log1p(yfull),      bins=80, alpha=0.6, color='steelblue', label='Train actual')\n"
    "axes[1].hist(np.log1p(pred_final), bins=80, alpha=0.6, color='coral',     label='Test pred')\n"
    "axes[1].set_title('log1p(demand): Train vs Final Predictions')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{OUT}/submissions/final_distribution_check.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# ── 14.7 Save submission ──────────────────────────────────────────────────
cells.append(code(
    "# 14.7 — Build and save final_submission.csv\n"
    "final_sub = pd.DataFrame({'Index': test_idx, 'demand': pred_final})\n"
    "\n"
    "# shape check\n"
    "assert final_sub.shape == (41778, 2), f'Wrong shape: {final_sub.shape}'\n"
    "assert list(final_sub.columns) == ['Index', 'demand'], f'Wrong columns: {final_sub.columns.tolist()}'\n"
    "assert final_sub.isnull().sum().sum() == 0, 'Nulls found!'\n"
    "\n"
    "sub_path = f'{OUT}/submissions/final_submission.csv'\n"
    "final_sub.to_csv(sub_path, index=False)\n"
    "print(f'Saved: {sub_path}')\n"
    "print(f'Shape: {final_sub.shape}')\n"
    "print(f'dtypes:\\n{final_sub.dtypes.to_string()}')\n"
    "print(f'Nulls : {final_sub.isnull().sum().sum()}')\n"
    "print(f'\\nFirst 10 rows:')\n"
    "print(final_sub.head(10).to_string(index=False))\n"
    "print(f'\\nLast 10 rows:')\n"
    "print(final_sub.tail(10).to_string(index=False))\n"
    "print(f'\\nSubmission stats:')\n"
    "print(final_sub['demand'].describe().to_string())"
))

# ── 14.8 SHAP on final LGB ────────────────────────────────────────────────
cells.append(code(
    "# 14.8 — SHAP feature importance from full-data LGB\n"
    "import shap\n"
    "print('Computing SHAP values (sample=3000 rows)...')\n"
    "shap_samp = Xfull.sample(min(3000, len(Xfull)), random_state=SEED)\n"
    "explainer  = shap.TreeExplainer(final_lgb)\n"
    "shap_vals  = explainer.shap_values(shap_samp)\n"
    "shap_imp   = pd.DataFrame({'feature': FEAT_COLS,\n"
    "                           'shap_importance': np.abs(shap_vals).mean(axis=0)})\\\n"
    "               .sort_values('shap_importance', ascending=False).reset_index(drop=True)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n"
    "sns.barplot(data=shap_imp.head(20), x='shap_importance', y='feature',\n"
    "            palette='viridis', ax=axes[0])\n"
    "axes[0].set_title('Final Model — Top 20 SHAP Feature Importance')\n"
    "axes[0].set_xlabel('mean(|SHAP|)')\n"
    "\n"
    "shap.summary_plot(shap_vals, shap_samp, plot_type='dot',\n"
    "                  max_display=15, show=False)\n"
    "axes[1].set_title('SHAP Dot Plot (top 15)')\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{OUT}/submissions/final_shap_importance.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('\\nTop 10 SHAP features:')\n"
    "print(shap_imp.head(10).to_string(index=False))\n"
    "shap_imp.to_csv(f'{OUT}/features/final_shap_importance.csv', index=False)"
))

# ── 14.9 Final report ─────────────────────────────────────────────────────
cells.append(code(
    "# 14.9 — Final Report\n"
    "print()\n"
    "print('╔' + '═'*62 + '╗')\n"
    "print('║' + ' FLIPKART TRAFFIC DEMAND PREDICTION — FINAL REPORT '.center(62) + '║')\n"
    "print('╠' + '═'*62 + '╣')\n"
    "\n"
    "print('║ PERFORMANCE'.ljust(63) + '║')\n"
    "print(f'║   Best CV R²          : {oof_r2:.6f}'.ljust(63) + '║')\n"
    "print(f'║   Competition score   : {100*oof_r2:.4f}  (target > 95.0)'.ljust(63) + '║')\n"
    "target_met = '✓ TARGET MET' if oof_r2 >= 0.95 else '✗ below target'\n"
    "print(f'║   Status              : {target_met}'.ljust(63) + '║')\n"
    "\n"
    "print('╠' + '═'*62 + '╣')\n"
    "print('║ FINAL ENSEMBLE (full-data retrain)'.ljust(63) + '║')\n"
    "print(f'║   LightGBM  weight={ridge_w[0]:.4f}   n_estimators=3000'.ljust(63) + '║')\n"
    "print(f'║   XGBoost   weight={ridge_w[1]:.4f}   n_estimators=3000'.ljust(63) + '║')\n"
    "print(f'║   CatBoost  weight={ridge_w[2]:.4f}   iterations=3000'.ljust(63) + '║')\n"
    "print(f'║   Intercept           : {ridge_i:.6f}'.ljust(63) + '║')\n"
    "print(f'║   Blending            : Ridge meta-learner (5-fold CV)'.ljust(63) + '║')\n"
    "\n"
    "print('╠' + '═'*62 + '╣')\n"
    "print('║ TOP 10 FEATURES (SHAP, final LGB)'.ljust(63) + '║')\n"
    "for i, row in shap_imp.head(10).iterrows():\n"
    "    line = f'║   {i+1:2d}. {row[\"feature\"]:<30} {row[\"shap_importance\"]:.5f}'\n"
    "    print(line.ljust(63) + '║')\n"
    "\n"
    "print('╠' + '═'*62 + '╣')\n"
    "print('║ KEY FEATURE ENGINEERING DECISIONS'.ljust(63) + '║')\n"
    "decisions = [\n"
    "    'OOF target encoding (geo×slot, geo×hour) — #1 signal',\n"
    "    'Geohash decode → lat/lon (spatial smoothing)',\n"
    "    'Cyclical time encoding (sin/cos) for hour/slot',\n"
    "    'KMeans geo clusters (k=20, best silhouette)',\n"
    "    'Road capacity score (lanes × large-vehicle flag)',\n"
    "    'Geohash neighbor mean/max demand (context)',\n"
    "    'Temperature imputation by geohash median',\n"
    "    'RoadType imputed from geohash mode (79.6% stable)',\n"
    "    'Frequency encoding (geohash, geohash×hour)',\n"
    "]\n"
    "for d in decisions:\n"
    "    print(f'║   • {d}'.ljust(63) + '║')\n"
    "\n"
    "print('╠' + '═'*62 + '╣')\n"
    "print('║ SUBMISSION FILE'.ljust(63) + '║')\n"
    "print(f'║   Path   : submissions/final_submission.csv'.ljust(63) + '║')\n"
    "print(f'║   Shape  : {final_sub.shape[0]} rows × {final_sub.shape[1]} cols'.ljust(63) + '║')\n"
    "print(f'║   Demand : min={pred_final.min():.4f}  max={pred_final.max():.4f}  '\n"
    "      f'mean={pred_final.mean():.4f}'.ljust(43) + '║')\n"
    "print(f'║   Nulls  : {final_sub.isnull().sum().sum()}'.ljust(63) + '║')\n"
    "print('╚' + '═'*62 + '╝')"
))

# ── inject ────────────────────────────────────────────────────────────────
with open(NB) as f:
    nb = json.load(f)

nb['cells'] = nb['cells'] + cells

seen = set()
for c in nb['cells']:
    v = uuid.uuid4().hex[:8]
    while v in seen: v = uuid.uuid4().hex[:8]
    c['id'] = v; seen.add(v)
    if c['cell_type'] == 'code' and not c.get('outputs'):
        c['outputs'] = []; c['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(cells)} cells. Total now: {len(nb['cells'])}")
