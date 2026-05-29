"""Replace Phase 4 (cells 46-55) with the comprehensive 7-group feature pipeline.
Ends by defining tr, te, X, y, X_test, FEATURES, TARGET so Phases 5-9 keep working."""
import json, uuid

NB = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/notebooks/flipkart_traffic_prediction.ipynb"

def nid():
    return uuid.uuid4().hex[:8]

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": nid(),
            "metadata": {}, "outputs": [], "source": src if isinstance(src, list) else [src]}

def md(src):
    return {"cell_type": "markdown", "id": nid(), "metadata": {},
            "source": src if isinstance(src, list) else [src]}

cells = []

# ── Phase 4 header ───────────────────────────────────────────────────────────
cells.append(md([
    "---\n",
    "## Phase 4 — Comprehensive Feature Engineering\n",
    "Pipeline fits **only on train**, transforms both train & test. Unseen geohashes fall back to global mean.\n",
    "\n",
    "| Group | Features |\n",
    "|-------|----------|\n",
    "| 1 — Temporal | hour/minute, cyclical sin/cos, peak/night flags, time-of-day bucket, day cyclical, weekend |\n",
    "| 2 — Geospatial | lat/lon, geohash-5/-4 truncation, distance-from-center, KMeans clusters (k chosen by silhouette) |\n",
    "| 3 — Target Encoding | **5-fold OOF** mean/median/std/max across 10 key/interaction groupings (leak-safe) |\n",
    "| 4 — Road / Infra | binary encodings, capacity & infrastructure scores, ordinal RoadType, lanes×roadtype |\n",
    "| 5 — Weather / Env | demand-ordered Weather ordinal, quantile temp bins, severity score, interactions |\n",
    "| 6 — Neighbor / Context | 8-neighbor mean/max demand, neighbor ratio |\n",
    "| 7 — Frequency | geohash, geohash×hour, weather, roadtype frequency |\n",
    "\n",
    "Outputs: `/features/train_features.csv`, `/features/test_features.csv`"
]))

# ── 4.0 fresh copies + shared OOF folds ──────────────────────────────────────
cells.append(code([
    "# 4.0 — Fresh working copies + shared OOF fold assignment\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "\n",
    "tr = train.copy()\n",
    "te = test.copy()\n",
    "TARGET = 'demand'\n",
    "\n",
    "# Folds reused later by the model CV (same seed -> identical splits -> leak-free target encoding)\n",
    "FE_FOLDS = 5\n",
    "fe_kf = KFold(n_splits=FE_FOLDS, shuffle=True, random_state=SEED)\n",
    "tr['_fold'] = -1\n",
    "for f, (_, val_idx) in enumerate(fe_kf.split(tr)):\n",
    "    tr.loc[tr.index[val_idx], '_fold'] = f\n",
    "print('Fold sizes:', tr['_fold'].value_counts().sort_index().tolist())"
]))

# ── GROUP 1 — Temporal ───────────────────────────────────────────────────────
cells.append(code([
    "# ══ GROUP 1 — TEMPORAL FEATURES ══\n",
    "def time_of_day_bucket(h):\n",
    "    if 5 <= h <= 11:  return 0   # morning\n",
    "    if 12 <= h <= 16: return 1   # afternoon\n",
    "    if 17 <= h <= 20: return 2   # evening\n",
    "    return 3                     # night\n",
    "\n",
    "for df in [tr, te]:\n",
    "    parts = df['timestamp'].str.split(':')\n",
    "    df['hour']   = parts.str[0].astype(int)\n",
    "    df['minute'] = parts.str[1].astype(int)\n",
    "    # NOTE: timestamp is 'H:MM' — no seconds component available\n",
    "    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15\n",
    "    # 2) cyclical encodings\n",
    "    df['hour_sin']   = np.sin(2 * np.pi * df['hour']   / 24)\n",
    "    df['hour_cos']   = np.cos(2 * np.pi * df['hour']   / 24)\n",
    "    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)\n",
    "    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)\n",
    "    # 3) peak-hour flag (7-9am, 5-8pm)\n",
    "    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)\n",
    "    # 4) night flag (11pm-5am)\n",
    "    df['is_night'] = df['hour'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)\n",
    "    # 5) time-of-day bucket\n",
    "    df['time_of_day'] = df['hour'].map(time_of_day_bucket)\n",
    "    # 6) day cyclical (day is an absolute index; %7 -> pseudo day-of-week)\n",
    "    df['day_of_week'] = df['day'] % 7\n",
    "    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)\n",
    "    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)\n",
    "    # 7) weekend flag\n",
    "    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)\n",
    "\n",
    "print('Group 1 done. day unique -> train:', sorted(tr['day'].unique()),\n",
    "      ' test:', sorted(te['day'].unique()))\n",
    "print('  (day is constant within each split -> day-derived cols are zero-variance in train,')\n",
    "print('   they are auto-dropped in the assembly step below.)')"
]))

# ── GROUP 2 — Geospatial ─────────────────────────────────────────────────────
cells.append(code([
    "# ══ GROUP 2 — GEOSPATIAL FEATURES ══\n",
    "import pygeohash as pgh\n",
    "\n",
    "# 8) decode geohash -> lat / lon\n",
    "all_geo = pd.unique(pd.concat([tr['geohash'], te['geohash']]))\n",
    "coords = {}\n",
    "for gh in all_geo:\n",
    "    d = pgh.decode_exactly(gh)\n",
    "    coords[gh] = (d.latitude, d.longitude)\n",
    "\n",
    "for df in [tr, te]:\n",
    "    df['lat'] = df['geohash'].map(lambda g: coords[g][0])\n",
    "    df['lon'] = df['geohash'].map(lambda g: coords[g][1])\n",
    "    # 9) / 10) coarser geohash truncations\n",
    "    df['geohash5'] = df['geohash'].str[:5]\n",
    "    df['geohash4'] = df['geohash'].str[:4]\n",
    "\n",
    "# 11) haversine distance from city center (train mean lat/lon as proxy)\n",
    "center_lat = tr['lat'].mean()\n",
    "center_lon = tr['lon'].mean()\n",
    "def haversine(lat, lon, clat, clon):\n",
    "    R = 6371.0; p = np.pi / 180\n",
    "    a = (0.5 - np.cos((clat - lat) * p) / 2\n",
    "         + np.cos(lat * p) * np.cos(clat * p) * (1 - np.cos((clon - lon) * p)) / 2)\n",
    "    return 2 * R * np.arcsin(np.sqrt(a))\n",
    "for df in [tr, te]:\n",
    "    df['dist_from_center'] = haversine(df['lat'].values, df['lon'].values, center_lat, center_lon)\n",
    "\n",
    "# 12) KMeans on unique geohash coords; pick k by silhouette\n",
    "geo_xy = pd.DataFrame([(g, coords[g][0], coords[g][1]) for g in all_geo],\n",
    "                      columns=['geohash', 'lat', 'lon'])\n",
    "best_k, best_sil, best_km = None, -1, None\n",
    "for k in [20, 50, 100]:\n",
    "    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)\n",
    "    labels = km.fit_predict(geo_xy[['lat', 'lon']])\n",
    "    sil = silhouette_score(geo_xy[['lat', 'lon']], labels)\n",
    "    print(f'  k={k:3d}  silhouette={sil:.4f}')\n",
    "    if sil > best_sil:\n",
    "        best_k, best_sil, best_km = k, sil, km\n",
    "print(f'-> chosen k={best_k} (silhouette={best_sil:.4f})')\n",
    "\n",
    "# 13) assign cluster label per geohash\n",
    "geo_xy['geo_cluster'] = best_km.predict(geo_xy[['lat', 'lon']])\n",
    "cluster_map = dict(zip(geo_xy['geohash'], geo_xy['geo_cluster']))\n",
    "for df in [tr, te]:\n",
    "    df['geo_cluster'] = df['geohash'].map(cluster_map).astype(int)\n",
    "\n",
    "# label-encode the truncated geohashes to integer codes (fit on combined categories)\n",
    "for col in ['geohash5', 'geohash4']:\n",
    "    cats = pd.Categorical(pd.concat([tr[col], te[col]]))\n",
    "    mapping = {c: i for i, c in enumerate(cats.categories)}\n",
    "    tr[col + '_enc'] = tr[col].map(mapping).astype(int)\n",
    "    te[col + '_enc'] = te[col].map(mapping).astype(int)\n",
    "print('Group 2 done.')"
]))

# ── GROUP 4 — Road & Infrastructure (before target encoding) ─────────────────
cells.append(code([
    "# ══ GROUP 4 — ROAD & INFRASTRUCTURE (run before target encoding) ══\n",
    "# Impute RoadType from geohash mode (EDA: 79.6% geohashes have a single RoadType)\n",
    "def geohash_mode_fill(col):\n",
    "    src = train.dropna(subset=[col])\n",
    "    mode_by_geo = src.groupby('geohash')[col].agg(lambda s: s.mode().iat[0])\n",
    "    global_mode = train[col].mode().iat[0]\n",
    "    for df in [tr, te]:\n",
    "        df[col] = df[col].fillna(df['geohash'].map(mode_by_geo)).fillna(global_mode)\n",
    "\n",
    "geohash_mode_fill('RoadType')\n",
    "geohash_mode_fill('Weather')\n",
    "\n",
    "# 24) binary encodings\n",
    "for df in [tr, te]:\n",
    "    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)\n",
    "    df['Landmarks_bin']     = (df['Landmarks'] == 'Yes').astype(int)\n",
    "    # 25) capacity / 26) infrastructure scores\n",
    "    df['road_capacity_score']   = df['NumberofLanes'] * (1 + df['LargeVehicles_bin'])\n",
    "    df['infrastructure_score']  = df['road_capacity_score'] + df['Landmarks_bin']\n",
    "\n",
    "# 27) RoadType ordinal — natural ordering exists (EDA: Highway >> Street >> Residential),\n",
    "#     derive the order data-driven from train mean demand\n",
    "road_order = tr.groupby('RoadType')[TARGET].mean().sort_values().index.tolist()\n",
    "road_map = {r: i for i, r in enumerate(road_order)}\n",
    "print('RoadType ordinal (by ascending mean demand):', road_map)\n",
    "for df in [tr, te]:\n",
    "    df['RoadType_ord'] = df['RoadType'].map(road_map).astype(int)\n",
    "    # 28) lanes x roadtype interaction\n",
    "    df['lanes_x_roadtype'] = df['NumberofLanes'] * df['RoadType_ord']\n",
    "print('Group 4 done.')"
]))

# ── GROUP 5 — Weather & Environment ──────────────────────────────────────────
cells.append(code([
    "# ══ GROUP 5 — WEATHER & ENVIRONMENT ══\n",
    "# Impute Temperature: geohash median -> global median (train-fit)\n",
    "geo_temp = train.groupby('geohash')['Temperature'].median()\n",
    "global_temp = train['Temperature'].median()\n",
    "for df in [tr, te]:\n",
    "    df['Temperature'] = df['Temperature'].fillna(df['geohash'].map(geo_temp)).fillna(global_temp)\n",
    "\n",
    "# 29) Weather ordinal by avg demand impact (data-driven, ascending)\n",
    "weather_order = tr.groupby('Weather')[TARGET].mean().sort_values().index.tolist()\n",
    "weather_map = {w: i for i, w in enumerate(weather_order)}\n",
    "print('Weather ordinal (by ascending mean demand):', weather_map)\n",
    "\n",
    "# 31) weather severity score (physical severity: clearer -> lower)\n",
    "severity_map = {'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3}\n",
    "\n",
    "# 30) temp quantile bins (5 buckets) — fit edges on train, apply to test\n",
    "tr['temp_binned'], temp_bins = pd.qcut(tr['Temperature'], 5, labels=False,\n",
    "                                        retbins=True, duplicates='drop')\n",
    "tr['temp_binned'] = tr['temp_binned'].astype(int)\n",
    "te['temp_binned'] = pd.cut(te['Temperature'], bins=temp_bins, labels=False,\n",
    "                            include_lowest=True)\n",
    "te['temp_binned'] = te['temp_binned'].fillna(-1).astype(int)  # outside train range\n",
    "\n",
    "for df in [tr, te]:\n",
    "    df['Weather_ord']            = df['Weather'].map(weather_map).astype(int)\n",
    "    df['weather_severity_score'] = df['Weather'].map(severity_map).astype(int)\n",
    "    # 32) weather x peak-hour, 33) temp x weather interactions\n",
    "    df['weather_x_peak_hour'] = df['weather_severity_score'] * df['is_peak_hour']\n",
    "    df['temp_x_weather']      = df['Temperature'] * df['weather_severity_score']\n",
    "print('Group 5 done.')"
]))

# ── GROUP 3 — OOF Target Encoding ────────────────────────────────────────────
cells.append(code([
    "# ══ GROUP 3 — TARGET ENCODING (5-fold out-of-fold, leak-safe) ══\n",
    "def oof_target_encode(group_cols, agg, new_col):\n",
    "    \"\"\"OOF encode train (per fold, excluding own fold); test uses full-train stats.\n",
    "    Unseen groups fall back to the global statistic.\"\"\"\n",
    "    g = tr[TARGET]\n",
    "    global_val = {'mean': g.mean(), 'median': g.median(),\n",
    "                  'std': g.std(), 'max': g.max(), 'min': g.min()}[agg]\n",
    "    # --- train OOF ---\n",
    "    tr[new_col] = np.nan\n",
    "    for f in range(FE_FOLDS):\n",
    "        bank = tr[tr['_fold'] != f]\n",
    "        stat = (bank.groupby(group_cols, observed=True)[TARGET]\n",
    "                    .agg(agg).rename(new_col).reset_index())\n",
    "        held = tr[tr['_fold'] == f][group_cols].reset_index(drop=True)\n",
    "        vals = held.merge(stat, on=group_cols, how='left')[new_col].values\n",
    "        tr.loc[tr['_fold'] == f, new_col] = vals\n",
    "    tr[new_col] = tr[new_col].fillna(global_val)\n",
    "    # --- test: full-train stats ---\n",
    "    stat_full = (tr.groupby(group_cols, observed=True)[TARGET]\n",
    "                   .agg(agg).rename(new_col).reset_index())\n",
    "    te[new_col] = (te[group_cols].merge(stat_full, on=group_cols, how='left')[new_col]\n",
    "                     .fillna(global_val).values)\n",
    "\n",
    "te_specs = [\n",
    "    (['geohash'],                'mean',   'te_geohash'),                  # 14\n",
    "    (['geohash', 'hour'],        'mean',   'te_geohash_hour'),             # 15\n",
    "    (['geohash', 'day'],         'mean',   'te_geohash_day'),              # 16\n",
    "    (['geohash', 'time_of_day'], 'mean',   'te_geohash_timeofday'),        # 17\n",
    "    (['RoadType', 'hour'],       'mean',   'te_roadtype_hour'),            # 18\n",
    "    (['Weather', 'hour'],        'mean',   'te_weather_hour'),             # 19\n",
    "    (['geohash4'],               'mean',   'te_geohash4'),                 # 20\n",
    "    (['geohash', 'hour'],        'median', 'te_median_geohash_hour'),      # 21\n",
    "    (['geohash', 'hour'],        'std',    'te_std_geohash_hour'),         # 22\n",
    "    (['geohash'],                'max',    'te_max_geohash'),              # 23\n",
    "]\n",
    "for cols, agg, name in te_specs:\n",
    "    oof_target_encode(cols, agg, name)\n",
    "    print(f'  {name:<26} <- {agg:<6} of demand by {cols}')\n",
    "print('Group 3 done (10 OOF target-encoded features).')"
]))

# ── GROUP 6 — Neighbor & Context ─────────────────────────────────────────────
cells.append(code([
    "# ══ GROUP 6 — NEIGHBOR & CONTEXT ══\n",
    "# 34) 8 neighbors per geohash via cell-center perturbation (robust across pygeohash versions)\n",
    "def get_neighbors(gh):\n",
    "    d = pgh.decode_exactly(gh)\n",
    "    p = len(gh)\n",
    "    out = []\n",
    "    for dla in (-1, 0, 1):\n",
    "        for dlo in (-1, 0, 1):\n",
    "            if dla == 0 and dlo == 0:\n",
    "                continue\n",
    "            nlat = d.latitude  + dla * 2 * d.latitude_error\n",
    "            nlon = d.longitude + dlo * 2 * d.longitude_error\n",
    "            out.append(pgh.encode(nlat, nlon, precision=p))\n",
    "    return out\n",
    "\n",
    "geo_mean_full = train.groupby('geohash')[TARGET].mean()  # full-train, train-only target\n",
    "gm = geo_mean_full.to_dict()\n",
    "global_mean = train[TARGET].mean()\n",
    "\n",
    "neigh_mean, neigh_max = {}, {}\n",
    "for gh in all_geo:\n",
    "    vals = [gm[n] for n in get_neighbors(gh) if n in gm]\n",
    "    if vals:\n",
    "        neigh_mean[gh], neigh_max[gh] = float(np.mean(vals)), float(np.max(vals))\n",
    "    else:\n",
    "        neigh_mean[gh], neigh_max[gh] = global_mean, global_mean\n",
    "\n",
    "for df in [tr, te]:\n",
    "    df['mean_neighbor_demand'] = df['geohash'].map(neigh_mean)          # 35\n",
    "    df['max_neighbor_demand']  = df['geohash'].map(neigh_max)           # 36\n",
    "    own = df['geohash'].map(gm).fillna(global_mean)\n",
    "    df['neighbor_demand_ratio'] = own / (df['mean_neighbor_demand'] + 1e-6)  # 37\n",
    "print('Group 6 done.')"
]))

# ── GROUP 7 — Frequency Encoding ─────────────────────────────────────────────
cells.append(code([
    "# ══ GROUP 7 — FREQUENCY ENCODING (counts from train; unseen -> 0) ══\n",
    "# 38) geohash frequency\n",
    "geo_freq = train['geohash'].value_counts()\n",
    "# 39) geohash x hour frequency (train, using parsed hour)\n",
    "_tr_hour = train['timestamp'].str.split(':').str[0].astype(int)\n",
    "gh_hour_freq = train.assign(_h=_tr_hour).groupby(['geohash', '_h']).size()\n",
    "gh_hour_freq = gh_hour_freq.rename('geohash_hour_frequency').reset_index()\n",
    "gh_hour_freq.columns = ['geohash', 'hour', 'geohash_hour_frequency']\n",
    "# 40) weather / roadtype frequency (use imputed values from tr to stay consistent)\n",
    "weather_freq  = tr.loc[tr['_fold'] >= -1, 'Weather'].value_counts()  # all train rows\n",
    "roadtype_freq = tr['RoadType'].value_counts()\n",
    "\n",
    "for df in [tr, te]:\n",
    "    df['geohash_frequency']  = df['geohash'].map(geo_freq).fillna(0).astype(int)\n",
    "    df['weather_frequency']  = df['Weather'].map(weather_freq).fillna(0).astype(int)\n",
    "    df['roadtype_frequency'] = df['RoadType'].map(roadtype_freq).fillna(0).astype(int)\n",
    "\n",
    "tr = tr.merge(gh_hour_freq, on=['geohash', 'hour'], how='left')\n",
    "te = te.merge(gh_hour_freq, on=['geohash', 'hour'], how='left')\n",
    "tr['geohash_hour_frequency'] = tr['geohash_hour_frequency'].fillna(0).astype(int)\n",
    "te['geohash_hour_frequency'] = te['geohash_hour_frequency'].fillna(0).astype(int)\n",
    "print('Group 7 done.')"
]))

# ── 4.FINAL — Assemble feature matrix, drop zero-variance, save, importance ──
cells.append(code([
    "# ══ ASSEMBLY — build X / X_test, drop zero-variance, save CSVs, importance preview ══\n",
    "FEATURES = [\n",
    "    # Group 1 — temporal\n",
    "    'hour', 'minute', 'time_slot', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',\n",
    "    'is_peak_hour', 'is_night', 'time_of_day', 'day_of_week', 'day_sin', 'day_cos', 'is_weekend',\n",
    "    # Group 2 — geospatial\n",
    "    'lat', 'lon', 'dist_from_center', 'geohash5_enc', 'geohash4_enc', 'geo_cluster',\n",
    "    # Group 3 — target encoding (OOF)\n",
    "    'te_geohash', 'te_geohash_hour', 'te_geohash_day', 'te_geohash_timeofday',\n",
    "    'te_roadtype_hour', 'te_weather_hour', 'te_geohash4',\n",
    "    'te_median_geohash_hour', 'te_std_geohash_hour', 'te_max_geohash',\n",
    "    # Group 4 — road / infra\n",
    "    'NumberofLanes', 'LargeVehicles_bin', 'Landmarks_bin',\n",
    "    'road_capacity_score', 'infrastructure_score', 'RoadType_ord', 'lanes_x_roadtype',\n",
    "    # Group 5 — weather / env\n",
    "    'Weather_ord', 'Temperature', 'temp_binned', 'weather_severity_score',\n",
    "    'weather_x_peak_hour', 'temp_x_weather',\n",
    "    # Group 6 — neighbor / context\n",
    "    'mean_neighbor_demand', 'max_neighbor_demand', 'neighbor_demand_ratio',\n",
    "    # Group 7 — frequency\n",
    "    'geohash_frequency', 'geohash_hour_frequency', 'weather_frequency', 'roadtype_frequency',\n",
    "]\n",
    "\n",
    "X      = tr[FEATURES].copy()\n",
    "y      = tr[TARGET].values\n",
    "X_test = te[FEATURES].copy()\n",
    "\n",
    "# residual NaN safety (train medians)\n",
    "med = X.median()\n",
    "X      = X.fillna(med)\n",
    "X_test = X_test.fillna(med)\n",
    "\n",
    "# drop zero-variance columns (e.g. day-derived cols constant because day is fixed per split)\n",
    "zero_var = [c for c in FEATURES if X[c].nunique() <= 1]\n",
    "if zero_var:\n",
    "    print('Dropping zero-variance columns:', zero_var)\n",
    "    FEATURES = [c for c in FEATURES if c not in zero_var]\n",
    "    X, X_test = X[FEATURES], X_test[FEATURES]\n",
    "\n",
    "print(f'\\nFinal feature count : {len(FEATURES)}')\n",
    "print(f'X shape             : {X.shape}')\n",
    "print(f'X_test shape        : {X_test.shape}')\n",
    "print(f'NaN in X / X_test   : {X.isnull().sum().sum()} / {X_test.isnull().sum().sum()}')\n",
    "\n",
    "# save feature sets\n",
    "train_out = X.copy(); train_out['demand'] = y; train_out.insert(0, 'Index', tr['Index'].values)\n",
    "test_out  = X_test.copy(); test_out.insert(0, 'Index', te['Index'].values)\n",
    "train_out.to_csv(f'{OUT}/features/train_features.csv', index=False)\n",
    "test_out.to_csv(f'{OUT}/features/test_features.csv', index=False)\n",
    "print(f'Saved -> {OUT}/features/train_features.csv  {train_out.shape}')\n",
    "print(f'Saved -> {OUT}/features/test_features.csv   {test_out.shape}')"
]))

cells.append(code([
    "# Feature importance preview — quick LightGBM on the assembled features\n",
    "_prev = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,\n",
    "                          random_state=SEED, n_jobs=-1, verbose=-1)\n",
    "_prev.fit(X, y)\n",
    "prev_imp = (pd.DataFrame({'feature': FEATURES, 'importance': _prev.feature_importances_})\n",
    "              .sort_values('importance', ascending=False).reset_index(drop=True))\n",
    "\n",
    "from sklearn.metrics import r2_score as _r2\n",
    "print(f'Quick in-sample R² (sanity, not CV): {_r2(y, _prev.predict(X)):.5f}\\n')\n",
    "print('Top 20 features by importance:')\n",
    "print(prev_imp.head(20).to_string(index=False))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 8))\n",
    "sns.barplot(data=prev_imp.head(20), x='importance', y='feature', palette='viridis', ax=ax)\n",
    "ax.set_title('Phase 4 Feature Importance Preview (top 20)')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{OUT}/features/phase4_importance_preview.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ─────────────────────────────────────────────────────────────────────────────
with open(NB) as f:
    nb = json.load(f)

# replace old Phase 4 (indices 46-55 inclusive) with new cells
old_lo, old_hi = 46, 55
nb['cells'] = nb['cells'][:old_lo] + cells + nb['cells'][old_hi + 1:]

# fresh unique ids + clear outputs everywhere
seen = set()
for c in nb['cells']:
    nidv = uuid.uuid4().hex[:8]
    while nidv in seen:
        nidv = uuid.uuid4().hex[:8]
    c['id'] = nidv; seen.add(nidv)
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Replaced cells {old_lo}-{old_hi} with {len(cells)} new cells.")
print(f"Total cells now: {len(nb['cells'])}")
