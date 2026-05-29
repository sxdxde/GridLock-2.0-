"""Add checkpointing to the modeling phases:
   - persistent Optuna studies (SQLite, resume completed trials)
   - per-fold model checkpoints during 5-fold OOF training (resume completed folds)
Patches cells 5.0 (setup), 6.1/7.1/8.1 (tuning) and the Phase 5 markdown by content marker."""
import json, uuid

NB = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/notebooks/flipkart_traffic_prediction.ipynb"

# ── new source for cell 5.0 (setup + checkpointed run_cv + resumable run_study) ──
SETUP = [
    "# 5.0 — Load engineered features + set up CV / metric / checkpointing\n",
    "from sklearn.model_selection import KFold, cross_val_predict\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.metrics import r2_score, mean_squared_error\n",
    "from scipy.optimize import minimize\n",
    "from optuna.trial import TrialState\n",
    "from optuna.samplers import TPESampler\n",
    "import optuna, joblib, time, glob\n",
    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
    "\n",
    "CKPT_DIR = f'{OUT}/models/checkpoints'\n",
    "os.makedirs(CKPT_DIR, exist_ok=True)\n",
    "\n",
    "# Set True to wipe checkpoints/studies and retune from scratch; False resumes where you left off.\n",
    "FRESH_START = False\n",
    "if FRESH_START:\n",
    "    for f in glob.glob(f'{CKPT_DIR}/*') + glob.glob(f'{OUT}/models/optuna_*.db'):\n",
    "        os.remove(f)\n",
    "    print('FRESH_START: cleared checkpoints and Optuna studies.')\n",
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
    "# ── Resumable Optuna study: trials persist in SQLite; reruns top up to n_target ──\n",
    "def run_study(name, objective, n_target):\n",
    "    storage = f'sqlite:///{OUT}/models/optuna_{name}.db'\n",
    "    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED),\n",
    "                                study_name=name, storage=storage, load_if_exists=True)\n",
    "    done = sum(t.state == TrialState.COMPLETE for t in study.trials)\n",
    "    remaining = max(0, n_target - done)\n",
    "    print(f'    [{name}] {done}/{n_target} trials complete; running {remaining} more')\n",
    "    if remaining:\n",
    "        study.optimize(objective, n_trials=remaining, callbacks=[make_cb(name)],\n",
    "                       show_progress_bar=False)\n",
    "    return study\n",
    "\n",
    "# ── Checkpointed 5-fold OOF: each finished fold is saved; reruns skip done folds ──\n",
    "def run_cv(make_model, fit_one, name, save=True):\n",
    "    ckpt = f'{CKPT_DIR}/cv_{name}.pkl'\n",
    "    if os.path.exists(ckpt):\n",
    "        st = joblib.load(ckpt)\n",
    "        oof, test_pred = st['oof'], st['test_pred']\n",
    "        scores, done, models = st['scores'], st['done'], st['models']\n",
    "        print(f'    [{name}] resume — folds done: {sorted(done)}')\n",
    "    else:\n",
    "        oof = np.zeros(len(X)); test_pred = np.zeros(len(X_test))\n",
    "        scores, done, models = {}, set(), []\n",
    "    for fold, (ti, vi) in enumerate(kf.split(X), 1):\n",
    "        if fold in done:\n",
    "            continue\n",
    "        m = make_model()\n",
    "        fit_one(m, X.iloc[ti], y[ti], X.iloc[vi], y[vi])\n",
    "        oof[vi]    = m.predict(X.iloc[vi])\n",
    "        test_pred += m.predict(X_test) / N_FOLDS\n",
    "        scores[fold] = r2_score(y[vi], oof[vi]); done.add(fold); models.append(m)\n",
    "        print(f'    [{name}] fold {fold}: R²={scores[fold]:.5f}  RMSE={rmse(y[vi], oof[vi]):.5f}')\n",
    "        joblib.dump({'oof': oof, 'test_pred': test_pred, 'scores': scores,\n",
    "                     'done': done, 'models': models}, ckpt)   # <-- checkpoint each fold\n",
    "    sc = [scores[f] for f in sorted(scores)]\n",
    "    print(f'    [{name}] OOF R²={r2_score(y, oof):.5f} | per-fold {np.mean(sc):.5f} ± {np.std(sc):.5f}')\n",
    "    if save:\n",
    "        joblib.dump(models, f'{OUT}/models/{name}_fold_models.pkl')\n",
    "    return oof, test_pred, sc, models",
]

# ── tuning cells: swap create_study/optimize for resumable run_study ──
LGB = [
    "# 6.1 — LightGBM Optuna tuning (resumable; single holdout, early stopping)\n",
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
    "study_lgb = run_study('lgb', lgb_objective, 150)\n",
    "print(f'\\nLGB best holdout R²={study_lgb.best_value:.5f}  ({time.time()-t0:.0f}s)')\n",
    "print('LGB best params:', study_lgb.best_params)\n",
    "\n",
    "best_lgb = dict(study_lgb.best_params)\n",
    "best_lgb.update(objective='regression', metric='rmse', bagging_fraction=0.8,\n",
    "                bagging_freq=5, n_estimators=3000, random_state=SEED, n_jobs=-1, verbose=-1)\n",
    "json.dump(best_lgb, open(f'{OUT}/models/best_lgb_params.json', 'w'), indent=2)",
]

XGB = [
    "# 7.1 — XGBoost Optuna tuning (resumable)\n",
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
    "study_xgb = run_study('xgb', xgb_objective, 100)\n",
    "print(f'\\nXGB best holdout R²={study_xgb.best_value:.5f}  ({time.time()-t0:.0f}s)')\n",
    "print('XGB best params:', study_xgb.best_params)\n",
    "\n",
    "best_xgb = dict(study_xgb.best_params)\n",
    "best_xgb.update(n_estimators=3000, tree_method='hist', random_state=SEED, n_jobs=-1,\n",
    "                early_stopping_rounds=100, eval_metric='rmse')\n",
    "json.dump(best_xgb, open(f'{OUT}/models/best_xgb_params.json', 'w'), indent=2)",
]

CAT = [
    "# 8.1 — CatBoost Optuna tuning (resumable; with cat_features)\n",
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
    "study_cat = run_study('cat', cat_objective, 100)\n",
    "print(f'\\nCAT best holdout R²={study_cat.best_value:.5f}  ({time.time()-t0:.0f}s)')\n",
    "print('CAT best params:', study_cat.best_params)\n",
    "\n",
    "best_cat = dict(study_cat.best_params)\n",
    "best_cat.update(iterations=3000, learning_rate=0.05, random_seed=SEED,\n",
    "                eval_metric='RMSE', od_type='Iter', od_wait=100, verbose=False)\n",
    "json.dump(best_cat, open(f'{OUT}/models/best_cat_params.json', 'w'), indent=2)",
]

MARK_APPEND = (
    "\n\n**Checkpointing (so a stopped run resumes, not restarts):** Optuna studies persist to "
    "`models/optuna_{lgb,xgb,cat}.db` (completed trials survive); each finished CV fold is saved to "
    "`models/checkpoints/cv_*.pkl`. Re-running a cell tops trials/folds back up to target. "
    "Set `FRESH_START=True` in cell 5.0 to wipe them and retune from scratch."
)

with open(NB) as f:
    nb = json.load(f)

patched = []
for c in nb['cells']:
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown' and 'Phase 5 — Modeling Setup' in src:
        c['source'] = (src + MARK_APPEND).splitlines(keepends=True)
        patched.append('md:Phase5')
    elif src.startswith('# 5.0 —'):
        c['source'] = SETUP; patched.append('5.0')
    elif src.startswith('# 6.1 —'):
        c['source'] = LGB; patched.append('6.1')
    elif src.startswith('# 7.1 —'):
        c['source'] = XGB; patched.append('7.1')
    elif src.startswith('# 8.1 —'):
        c['source'] = CAT; patched.append('8.1')

# clear outputs everywhere + ensure unique ids
seen = set()
for c in nb['cells']:
    v = uuid.uuid4().hex[:8]
    while v in seen: v = uuid.uuid4().hex[:8]
    c['id'] = v; seen.add(v)
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print("Patched:", patched)
