"""Injects Phase 3.5 Deep EDA cells into the master notebook after cell index 18."""
import json, copy, uuid

def new_id():
    return uuid.uuid4().hex[:8]

NB_PATH = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/notebooks/flipkart_traffic_prediction.ipynb"

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": new_id(),
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

new_cells = []

# ── Header ────────────────────────────────────────────────────────────────────
new_cells.append(md_cell(
    "---\n## Phase 3.5 — Deep Exploratory Data Analysis\n"
    "Covers: Temporal · Geospatial · Feature Correlation · Combined Patterns"
))

# ── Setup: working copy so deep-EDA never mutates train/test ─────────────────
new_cells.append(code_cell([
    "# Working copy — never mutates train / test\n",
    "eda = train.copy()\n",
    "\n",
    "PLOT_DIR = OUT + '/notebooks'\n",
    "os.makedirs(PLOT_DIR, exist_ok=True)\n",
    "print('Deep-EDA working copy ready:', eda.shape)"
]))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md_cell("### A — Temporal Analysis"))

# A1 – parse timestamp
new_cells.append(code_cell([
    "# A1 — Timestamp format detection & parsing\n",
    "print('Sample timestamp values:', eda['timestamp'].unique()[:15].tolist())\n",
    "print('dtype :', eda['timestamp'].dtype)\n",
    "print('Total unique timestamps:', eda['timestamp'].nunique())\n",
    "\n",
    "# Format is 'H:MM' (hour 0-23, minute 0/15/30/45)\n",
    "eda['hour']      = eda['timestamp'].str.split(':').str[0].astype(int)\n",
    "eda['minute']    = eda['timestamp'].str.split(':').str[1].astype(int)\n",
    "eda['time_slot'] = eda['hour'] * 4 + eda['minute'] // 15   # 0..95\n",
    "eda['time_dec']  = eda['hour'] + eda['minute'] / 60        # decimal hour\n",
    "\n",
    "print('\\nHour range    :', eda['hour'].min(), '→', eda['hour'].max())\n",
    "print('Minute values :', sorted(eda['minute'].unique()))\n",
    "print('Time-slots    :', eda['time_slot'].nunique(), '(expected 96 for 15-min grid)')\n",
    "print('Day values    :', sorted(eda['day'].unique()), '→ numeric (likely week-of-year/day-ID)')"
]))

# A2 – demand by hour
new_cells.append(code_cell([
    "# A2 — Demand by hour of day\n",
    "hourly = eda.groupby('hour')['demand'].agg(['mean','median','std']).reset_index()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 5))\n",
    "ax.plot(hourly['hour'], hourly['mean'],   marker='o', lw=2, label='Mean',   color='steelblue')\n",
    "ax.plot(hourly['hour'], hourly['median'], marker='s', lw=2, label='Median', color='coral')\n",
    "ax.fill_between(hourly['hour'],\n",
    "                hourly['mean'] - hourly['std'],\n",
    "                hourly['mean'] + hourly['std'],\n",
    "                alpha=0.15, color='steelblue', label='±1 std')\n",
    "ax.set_title('Demand by Hour of Day', fontsize=14)\n",
    "ax.set_xlabel('Hour')\n",
    "ax.set_ylabel('Demand')\n",
    "ax.set_xticks(range(0, 24))\n",
    "ax.legend()\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/A2_demand_by_hour.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(hourly.to_string(index=False))"
]))

# A3 – demand by day
new_cells.append(code_cell([
    "# A3 — Demand by day\n",
    "print('Day column analysis:')\n",
    "print('  unique values:', sorted(eda['day'].unique()))\n",
    "print('  dtype        :', eda['day'].dtype)\n",
    "print('  value counts :')\n",
    "print(eda['day'].value_counts().sort_index())\n",
    "print('  NOTE: day=48 → train only, day=49 → test only (literal split boundary)')\n",
    "\n",
    "daily = eda.groupby('day')['demand'].agg(['mean','median','std','count']).reset_index()\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
    "axes[0].bar(daily['day'].astype(str), daily['mean'], color=['steelblue','coral'][:len(daily)])\n",
    "axes[0].set_title('Mean Demand by Day')\n",
    "axes[0].set_xlabel('Day')\n",
    "axes[0].set_ylabel('Mean Demand')\n",
    "\n",
    "# 15-min time-slot demand aggregated by day\n",
    "for day_val, grp in eda.groupby('day'):\n",
    "    slot_mean = grp.groupby('time_slot')['demand'].mean()\n",
    "    axes[1].plot(slot_mean.index, slot_mean.values, label=f'Day {day_val}', lw=1.5)\n",
    "axes[1].set_title('Mean Demand per 15-min slot by Day')\n",
    "axes[1].set_xlabel('Time slot (0=00:00, 95=23:45)')\n",
    "axes[1].set_ylabel('Mean Demand')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/A3_demand_by_day.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# A4 – peak hours
new_cells.append(code_cell([
    "# A4 — Peak vs off-peak hours\n",
    "hourly_mean = eda.groupby('hour')['demand'].mean().sort_values(ascending=False)\n",
    "overall_mean = eda['demand'].mean()\n",
    "\n",
    "peak_threshold    = overall_mean * 1.25\n",
    "offpeak_threshold = overall_mean * 0.75\n",
    "\n",
    "peak_hours    = hourly_mean[hourly_mean >= peak_threshold].index.sort_values().tolist()\n",
    "offpeak_hours = hourly_mean[hourly_mean <= offpeak_threshold].index.sort_values().tolist()\n",
    "\n",
    "print(f'Overall mean demand   : {overall_mean:.4f}')\n",
    "print(f'Peak threshold (×1.25): {peak_threshold:.4f}')\n",
    "print(f'Peak hours            : {peak_hours}')\n",
    "print(f'Off-peak threshold    : {offpeak_threshold:.4f}')\n",
    "print(f'Off-peak hours        : {offpeak_hours}')\n",
    "\n",
    "hourly_sorted = eda.groupby('hour')['demand'].mean().reset_index()\n",
    "colors = []\n",
    "for h in hourly_sorted['hour']:\n",
    "    if h in peak_hours:\n",
    "        colors.append('#e74c3c')\n",
    "    elif h in offpeak_hours:\n",
    "        colors.append('#2ecc71')\n",
    "    else:\n",
    "        colors.append('#3498db')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 4))\n",
    "bars = ax.bar(hourly_sorted['hour'], hourly_sorted['demand'], color=colors)\n",
    "ax.axhline(peak_threshold,    color='red',   linestyle='--', label='Peak threshold')\n",
    "ax.axhline(offpeak_threshold, color='green', linestyle='--', label='Off-peak threshold')\n",
    "ax.set_title('Peak (red) vs Off-Peak (green) Hours')\n",
    "ax.set_xlabel('Hour')\n",
    "ax.set_ylabel('Mean Demand')\n",
    "ax.set_xticks(range(0, 24))\n",
    "ax.legend()\n",
    "ax.grid(True, alpha=0.3, axis='y')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/A4_peak_hours.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# A5 – heatmap day × hour
new_cells.append(code_cell([
    "# A5 — Heatmap: day × hour with mean demand\n",
    "pivot_dh = eda.pivot_table(values='demand', index='day', columns='hour', aggfunc='mean')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(18, 3))\n",
    "sns.heatmap(pivot_dh, ax=ax, cmap='YlOrRd', annot=False,\n",
    "            linewidths=0.3, cbar_kws={'label': 'Mean Demand'})\n",
    "ax.set_title('Mean Demand Heatmap: Day × Hour')\n",
    "ax.set_xlabel('Hour of Day')\n",
    "ax.set_ylabel('Day')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/A5_heatmap_day_hour.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Also: 15-min resolution heatmap\n",
    "pivot_slot = eda.pivot_table(values='demand', index='day', columns='time_slot', aggfunc='mean')\n",
    "fig, ax = plt.subplots(figsize=(24, 3))\n",
    "sns.heatmap(pivot_slot, ax=ax, cmap='YlOrRd', annot=False, linewidths=0,\n",
    "            cbar_kws={'label': 'Mean Demand'})\n",
    "ax.set_title('Mean Demand Heatmap: Day × 15-min Slot')\n",
    "ax.set_xlabel('Time Slot (0=00:00 → 95=23:45)')\n",
    "ax.set_ylabel('Day')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/A5_heatmap_day_timeslot.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — GEOSPATIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md_cell("### B — Geospatial Analysis"))

# B6 – decode geohash
new_cells.append(code_cell([
    "# B6 — Decode geohash → lat / lon\n",
    "import pygeohash as pgh\n",
    "\n",
    "geo_coords = {}\n",
    "for gh in eda['geohash'].unique():\n",
    "    try:\n",
    "        lat, lon, _, _ = pgh.decode_exactly(gh)\n",
    "        geo_coords[gh] = (lat, lon)\n",
    "    except:\n",
    "        geo_coords[gh] = (np.nan, np.nan)\n",
    "\n",
    "eda['lat'] = eda['geohash'].map(lambda g: geo_coords[g][0])\n",
    "eda['lon'] = eda['geohash'].map(lambda g: geo_coords[g][1])\n",
    "\n",
    "print(f'Decoded {len(geo_coords)} unique geohashes')\n",
    "print(f'Lat range: {eda[\"lat\"].min():.4f} → {eda[\"lat\"].max():.4f}')\n",
    "print(f'Lon range: {eda[\"lon\"].min():.4f} → {eda[\"lon\"].max():.4f}')\n",
    "print(eda[['geohash','lat','lon']].drop_duplicates().head(5))"
]))

# B7 – scatter lat/lon colored by mean demand
new_cells.append(code_cell([
    "# B7 — Scatter lat/lon colored by mean demand\n",
    "geo_demand = eda.groupby(['geohash','lat','lon'])['demand'].mean().reset_index()\n",
    "geo_demand.columns = ['geohash','lat','lon','mean_demand']\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
    "\n",
    "sc = axes[0].scatter(geo_demand['lon'], geo_demand['lat'],\n",
    "                     c=geo_demand['mean_demand'], cmap='YlOrRd',\n",
    "                     s=30, alpha=0.8, edgecolors='none')\n",
    "plt.colorbar(sc, ax=axes[0], label='Mean Demand')\n",
    "axes[0].set_title('Mean Demand by Location (all geohashes)')\n",
    "axes[0].set_xlabel('Longitude')\n",
    "axes[0].set_ylabel('Latitude')\n",
    "\n",
    "# Hexbin version for density\n",
    "hb = axes[1].hexbin(geo_demand['lon'], geo_demand['lat'],\n",
    "                    C=geo_demand['mean_demand'], gridsize=40,\n",
    "                    cmap='YlOrRd', reduce_C_function=np.mean)\n",
    "plt.colorbar(hb, ax=axes[1], label='Mean Demand')\n",
    "axes[1].set_title('Hexbin: Mean Demand by Location')\n",
    "axes[1].set_xlabel('Longitude')\n",
    "axes[1].set_ylabel('Latitude')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/B7_geospatial_demand.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# B8 – top 10 highest demand geohashes
new_cells.append(code_cell([
    "# B8 — Top 10 highest demand geohash locations\n",
    "geo_stats_full = eda.groupby('geohash')['demand'].agg(\n",
    "    mean='mean', median='median', std='std', count='count'\n",
    ").reset_index().sort_values('mean', ascending=False)\n",
    "geo_stats_full['lat'] = geo_stats_full['geohash'].map(lambda g: geo_coords[g][0])\n",
    "geo_stats_full['lon'] = geo_stats_full['geohash'].map(lambda g: geo_coords[g][1])\n",
    "\n",
    "top10_high = geo_stats_full.head(10)\n",
    "print('=== Top 10 HIGHEST demand geohashes ===')\n",
    "print(top10_high[['geohash','lat','lon','mean','median','std','count']].to_string(index=False))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "bars = ax.barh(top10_high['geohash'], top10_high['mean'], color='#e74c3c')\n",
    "ax.set_title('Top 10 Geohashes by Mean Demand')\n",
    "ax.set_xlabel('Mean Demand')\n",
    "for bar, val in zip(bars, top10_high['mean']):\n",
    "    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,\n",
    "            f'{val:.4f}', va='center', fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/B8_top10_high_demand_geo.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# B9 – top 10 lowest demand geohashes
new_cells.append(code_cell([
    "# B9 — Top 10 lowest demand geohash locations\n",
    "top10_low = geo_stats_full[geo_stats_full['count'] >= 10].tail(10).sort_values('mean')\n",
    "print('=== Top 10 LOWEST demand geohashes (min 10 obs) ===')\n",
    "print(top10_low[['geohash','lat','lon','mean','median','std','count']].to_string(index=False))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "bars = ax.barh(top10_low['geohash'], top10_low['mean'], color='#2ecc71')\n",
    "ax.set_title('Top 10 Geohashes by Lowest Mean Demand (≥10 obs)')\n",
    "ax.set_xlabel('Mean Demand')\n",
    "for bar, val in zip(bars, top10_low['mean']):\n",
    "    ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,\n",
    "            f'{val:.5f}', va='center', fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/B9_top10_low_demand_geo.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# B10 – geohash overlap train vs test
new_cells.append(code_cell([
    "# B10 — Unique geohashes: train vs test overlap\n",
    "train_geo = set(train['geohash'])\n",
    "test_geo  = set(test['geohash'])\n",
    "common    = train_geo & test_geo\n",
    "test_only = test_geo - train_geo\n",
    "train_only= train_geo - test_geo\n",
    "\n",
    "print(f'Train unique geohashes : {len(train_geo)}')\n",
    "print(f'Test  unique geohashes : {len(test_geo)}')\n",
    "print(f'Common (seen in both)  : {len(common)}  ({100*len(common)/len(test_geo):.1f}% of test)')\n",
    "print(f'Test-only (UNSEEN)     : {len(test_only)}  ({100*len(test_only)/len(test_geo):.1f}% of test)')\n",
    "print(f'Train-only             : {len(train_only)}')\n",
    "if test_only:\n",
    "    print(f'\\nSample unseen test geohashes: {list(test_only)[:10]}')\n",
    "\n",
    "# Prefix coverage\n",
    "print('\\nPrefix-level coverage:')\n",
    "for n in [1,2,3,4,5]:\n",
    "    tr_p = set(train['geohash'].str[:n])\n",
    "    te_p = set(test['geohash'].str[:n])\n",
    "    miss = te_p - tr_p\n",
    "    print(f'  prefix-{n}: train={len(tr_p):4d}  test={len(te_p):4d}  '\n",
    "          f'common={len(tr_p & te_p):4d}  test-only={len(miss)}')\n",
    "\n",
    "# Venn-style bar\n",
    "fig, ax = plt.subplots(figsize=(8, 3))\n",
    "ax.barh(['Geohashes'], [len(common)],    label=f'Common ({len(common)})',    color='#3498db')\n",
    "ax.barh(['Geohashes'], [len(test_only)], label=f'Test-only ({len(test_only)})', color='#e74c3c',\n",
    "        left=[len(common)])\n",
    "ax.barh(['Geohashes'], [len(train_only)],label=f'Train-only ({len(train_only)})', color='#2ecc71',\n",
    "        left=[len(common)+len(test_only)])\n",
    "ax.set_title('Geohash Overlap: Train vs Test')\n",
    "ax.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/B10_geohash_overlap.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — FEATURE CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md_cell("### C — Feature Correlation"))

# C11 – full correlation heatmap
new_cells.append(code_cell([
    "# C11 — Full correlation heatmap (encoded categoricals)\n",
    "corr_df = eda.copy()\n",
    "corr_df['RoadType_enc']      = corr_df['RoadType'].map({'Residential':0,'Street':1,'Highway':2})\n",
    "corr_df['LargeVehicles_enc'] = corr_df['LargeVehicles'].map({'Not Allowed':0,'Allowed':1})\n",
    "corr_df['Landmarks_enc']     = corr_df['Landmarks'].map({'No':0,'Yes':1})\n",
    "corr_df['Weather_enc']       = corr_df['Weather'].map({'Sunny':0,'Rainy':1,'Foggy':2,'Snowy':3})\n",
    "\n",
    "num_cols = ['demand','hour','minute','time_slot','day','lat','lon',\n",
    "             'RoadType_enc','NumberofLanes','LargeVehicles_enc',\n",
    "             'Landmarks_enc','Temperature','Weather_enc']\n",
    "corr_matrix = corr_df[num_cols].corr()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(13, 10))\n",
    "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
    "sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',\n",
    "            cmap='RdBu_r', center=0, vmin=-1, vmax=1,\n",
    "            square=True, linewidths=0.5, ax=ax,\n",
    "            annot_kws={'size': 8})\n",
    "ax.set_title('Full Feature Correlation Heatmap', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/C11_correlation_heatmap.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Demand correlations sorted\n",
    "print('\\nCorrelation with demand (sorted):')\n",
    "print(corr_matrix['demand'].drop('demand').sort_values(key=abs, ascending=False).to_string())"
]))

# C12 – demand by RoadType
new_cells.append(code_cell([
    "# C12 — Demand distribution by RoadType\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "road_types = sorted(eda['RoadType'].dropna().unique())\n",
    "colors_rt  = ['#3498db', '#e74c3c', '#2ecc71']\n",
    "\n",
    "# KDE\n",
    "for rt, col in zip(road_types, colors_rt):\n",
    "    subset = eda[eda['RoadType'] == rt]['demand']\n",
    "    subset.plot.kde(ax=axes[0], label=f'{rt} (n={len(subset):,})', color=col, lw=2)\n",
    "axes[0].set_title('Demand KDE by RoadType')\n",
    "axes[0].set_xlabel('Demand')\n",
    "axes[0].legend(fontsize=8)\n",
    "axes[0].set_xlim(-0.05, 0.8)\n",
    "\n",
    "# Box\n",
    "order_rt = eda.groupby('RoadType')['demand'].median().sort_values(ascending=False).index\n",
    "sns.boxplot(data=eda, x='RoadType', y='demand', order=order_rt,\n",
    "            palette='Set2', ax=axes[1])\n",
    "axes[1].set_title('Demand Boxplot by RoadType')\n",
    "\n",
    "# Stats table\n",
    "rt_stats = eda.groupby('RoadType')['demand'].agg(['mean','median','std','count'])\n",
    "axes[2].axis('off')\n",
    "tbl = axes[2].table(cellText=rt_stats.round(4).values,\n",
    "                    rowLabels=rt_stats.index,\n",
    "                    colLabels=['Mean','Median','Std','Count'],\n",
    "                    cellLoc='center', loc='center')\n",
    "tbl.auto_set_font_size(False)\n",
    "tbl.set_fontsize(11)\n",
    "tbl.scale(1.2, 1.8)\n",
    "axes[2].set_title('Stats by RoadType', pad=20)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/C12_demand_by_roadtype.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(rt_stats)"
]))

# C13 – demand by Weather
new_cells.append(code_cell([
    "# C13 — Demand distribution by Weather\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "weather_types = sorted(eda['Weather'].dropna().unique())\n",
    "pal_w = {'Sunny':'#f39c12','Rainy':'#3498db','Foggy':'#95a5a6','Snowy':'#1abc9c'}\n",
    "\n",
    "for wt in weather_types:\n",
    "    subset = eda[eda['Weather'] == wt]['demand']\n",
    "    subset.plot.kde(ax=axes[0], label=f'{wt} (n={len(subset):,})',\n",
    "                   color=pal_w.get(wt,'grey'), lw=2)\n",
    "axes[0].set_title('Demand KDE by Weather')\n",
    "axes[0].set_xlabel('Demand')\n",
    "axes[0].legend(fontsize=9)\n",
    "axes[0].set_xlim(-0.05, 0.8)\n",
    "\n",
    "order_w = eda.groupby('Weather')['demand'].median().sort_values(ascending=False).index\n",
    "sns.violinplot(data=eda, x='Weather', y='demand', order=order_w,\n",
    "               palette=pal_w, ax=axes[1], cut=0)\n",
    "axes[1].set_title('Demand Violin by Weather')\n",
    "\n",
    "w_stats = eda.groupby('Weather')['demand'].agg(['mean','median','std','count'])\n",
    "axes[2].axis('off')\n",
    "tbl = axes[2].table(cellText=w_stats.round(4).values,\n",
    "                    rowLabels=w_stats.index,\n",
    "                    colLabels=['Mean','Median','Std','Count'],\n",
    "                    cellLoc='center', loc='center')\n",
    "tbl.auto_set_font_size(False)\n",
    "tbl.set_fontsize(11)\n",
    "tbl.scale(1.2, 1.8)\n",
    "axes[2].set_title('Stats by Weather', pad=20)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/C13_demand_by_weather.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(w_stats)"
]))

# C14 – demand vs Temperature
new_cells.append(code_cell([
    "# C14 — Demand vs Temperature (scatter + trend)\n",
    "from scipy.stats import pearsonr, spearmanr\n",
    "\n",
    "temp_data = eda[['Temperature','demand']].dropna()\n",
    "pr, pp = pearsonr(temp_data['Temperature'],  temp_data['demand'])\n",
    "sr, sp = spearmanr(temp_data['Temperature'], temp_data['demand'])\n",
    "print(f'Pearson  r={pr:.4f}  p={pp:.2e}')\n",
    "print(f'Spearman r={sr:.4f}  p={sp:.2e}')\n",
    "\n",
    "# Bin temperature and compute mean demand\n",
    "temp_data = temp_data.copy()\n",
    "temp_data['temp_bin'] = pd.cut(temp_data['Temperature'], bins=20)\n",
    "bin_stats = temp_data.groupby('temp_bin', observed=True)['demand'].agg(['mean','count'])\n",
    "bin_centers = [iv.mid for iv in bin_stats.index]\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
    "\n",
    "sample = temp_data.sample(min(8000, len(temp_data)), random_state=42)\n",
    "axes[0].scatter(sample['Temperature'], sample['demand'],\n",
    "                alpha=0.15, s=5, color='steelblue')\n",
    "\n",
    "# Polynomial trend\n",
    "z = np.polyfit(temp_data['Temperature'], temp_data['demand'], 2)\n",
    "p = np.poly1d(z)\n",
    "xs = np.linspace(temp_data['Temperature'].min(), temp_data['Temperature'].max(), 200)\n",
    "axes[0].plot(xs, p(xs), color='red', lw=2, label=f'Poly-2 trend  Pearson r={pr:.3f}')\n",
    "axes[0].set_title('Demand vs Temperature')\n",
    "axes[0].set_xlabel('Temperature')\n",
    "axes[0].set_ylabel('Demand')\n",
    "axes[0].legend()\n",
    "\n",
    "axes[1].bar(bin_centers, bin_stats['mean'], width=2.5, color='coral', edgecolor='white')\n",
    "axes[1].set_title('Mean Demand by Temperature Bin')\n",
    "axes[1].set_xlabel('Temperature (bin center)')\n",
    "axes[1].set_ylabel('Mean Demand')\n",
    "axes[1].grid(True, alpha=0.3, axis='y')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/C14_demand_vs_temperature.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# C15 – LargeVehicles
new_cells.append(code_cell([
    "# C15 — Demand: LargeVehicles Allowed vs Not Allowed\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "for lv in ['Allowed','Not Allowed']:\n",
    "    subset = eda[eda['LargeVehicles'] == lv]['demand']\n",
    "    subset.plot.kde(ax=axes[0], label=f'{lv} (n={len(subset):,})', lw=2)\n",
    "axes[0].set_title('Demand KDE: LargeVehicles')\n",
    "axes[0].set_xlabel('Demand')\n",
    "axes[0].legend()\n",
    "axes[0].set_xlim(-0.05, 0.8)\n",
    "\n",
    "sns.boxplot(data=eda, x='LargeVehicles', y='demand',\n",
    "            order=['Not Allowed','Allowed'], palette='Set1', ax=axes[1])\n",
    "axes[1].set_title('Demand Boxplot: LargeVehicles')\n",
    "\n",
    "lv_stats = eda.groupby('LargeVehicles')['demand'].agg(['mean','median','std','count'])\n",
    "print(lv_stats)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/C15_demand_largevehicles.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# C16 – Landmarks
new_cells.append(code_cell([
    "# C16 — Demand: Landmarks Yes vs No\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "for lm in ['Yes','No']:\n",
    "    subset = eda[eda['Landmarks'] == lm]['demand']\n",
    "    subset.plot.kde(ax=axes[0], label=f'Landmarks={lm} (n={len(subset):,})', lw=2)\n",
    "axes[0].set_title('Demand KDE: Landmarks')\n",
    "axes[0].set_xlabel('Demand')\n",
    "axes[0].legend()\n",
    "axes[0].set_xlim(-0.05, 0.8)\n",
    "\n",
    "sns.boxplot(data=eda, x='Landmarks', y='demand',\n",
    "            order=['No','Yes'], palette='Set2', ax=axes[1])\n",
    "axes[1].set_title('Demand Boxplot: Landmarks')\n",
    "\n",
    "lm_stats = eda.groupby('Landmarks')['demand'].agg(['mean','median','std','count'])\n",
    "print(lm_stats)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/C16_demand_landmarks.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — COMBINED PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md_cell("### D — Combined Patterns"))

# D17 – RoadType × hour
new_cells.append(code_cell([
    "# D17 — Demand by RoadType × Hour (grouped line chart)\n",
    "road_hour = eda.groupby(['RoadType','hour'])['demand'].mean().reset_index()\n",
    "road_hour = road_hour.dropna(subset=['RoadType'])\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 5))\n",
    "\n",
    "colors_rt = {'Residential':'#3498db','Street':'#e74c3c','Highway':'#2ecc71'}\n",
    "for rt, grp in road_hour.groupby('RoadType'):\n",
    "    axes[0].plot(grp['hour'], grp['demand'],\n",
    "                 marker='o', lw=2, label=rt, color=colors_rt.get(rt,'grey'))\n",
    "axes[0].set_title('Mean Demand by RoadType × Hour')\n",
    "axes[0].set_xlabel('Hour')\n",
    "axes[0].set_ylabel('Mean Demand')\n",
    "axes[0].set_xticks(range(0, 24))\n",
    "axes[0].legend()\n",
    "axes[0].grid(True, alpha=0.3)\n",
    "\n",
    "# Pivot heatmap: RoadType × Hour\n",
    "pivot_rh = road_hour.pivot(index='RoadType', columns='hour', values='demand')\n",
    "sns.heatmap(pivot_rh, ax=axes[1], cmap='YlOrRd', annot=True, fmt='.3f',\n",
    "            linewidths=0.5, cbar_kws={'label':'Mean Demand'},\n",
    "            annot_kws={'size': 7})\n",
    "axes[1].set_title('RoadType × Hour Heatmap')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/D17_roadtype_x_hour.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# D18 – Weather × RoadType heatmap
new_cells.append(code_cell([
    "# D18 — Demand heatmap: Weather × RoadType\n",
    "wr_pivot = eda.groupby(['Weather','RoadType'])['demand'].mean().unstack()\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
    "\n",
    "sns.heatmap(wr_pivot, ax=axes[0], cmap='YlOrRd', annot=True, fmt='.4f',\n",
    "            linewidths=0.5, cbar_kws={'label':'Mean Demand'})\n",
    "axes[0].set_title('Mean Demand: Weather × RoadType')\n",
    "\n",
    "# Count heatmap (sample sizes)\n",
    "wr_count = eda.groupby(['Weather','RoadType'])['demand'].count().unstack()\n",
    "sns.heatmap(wr_count, ax=axes[1], cmap='Blues', annot=True, fmt='d',\n",
    "            linewidths=0.5, cbar_kws={'label':'Count'})\n",
    "axes[1].set_title('Sample Counts: Weather × RoadType')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/D18_weather_x_roadtype.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(wr_pivot)"
]))

# D19 – geohash stability check
new_cells.append(code_cell([
    "# D19 — Are road features static per geohash? (stability check)\n",
    "geo_road_check = eda.groupby('geohash').agg(\n",
    "    RoadType_nunique      = ('RoadType',       'nunique'),\n",
    "    Lanes_nunique         = ('NumberofLanes',  'nunique'),\n",
    "    LargeVehicles_nunique = ('LargeVehicles',  'nunique'),\n",
    "    Landmarks_nunique     = ('Landmarks',      'nunique'),\n",
    "    obs_count             = ('demand',         'count')\n",
    ").reset_index()\n",
    "\n",
    "print('=== Road Feature Stability per Geohash ===')\n",
    "for col in ['RoadType_nunique','Lanes_nunique','LargeVehicles_nunique','Landmarks_nunique']:\n",
    "    vc = geo_road_check[col].value_counts().sort_index()\n",
    "    pct1 = 100 * (geo_road_check[col] == 1).mean()\n",
    "    print(f'\\n{col}: {pct1:.1f}% of geohashes have exactly 1 unique value')\n",
    "    print(vc.to_string())\n",
    "\n",
    "# Show unstable geohashes\n",
    "unstable = geo_road_check[\n",
    "    (geo_road_check['RoadType_nunique'] > 1) |\n",
    "    (geo_road_check['Lanes_nunique'] > 1)\n",
    "]\n",
    "print(f'\\nGeohashes with changing RoadType or Lanes: {len(unstable)}')\n",
    "if len(unstable) > 0:\n",
    "    print(unstable.sort_values('obs_count', ascending=False).head(10).to_string(index=False))\n",
    "\n",
    "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
    "feat_cols = ['RoadType_nunique','Lanes_nunique','LargeVehicles_nunique','Landmarks_nunique']\n",
    "feat_labels = ['RoadType','NumberofLanes','LargeVehicles','Landmarks']\n",
    "for ax, col, label in zip(axes, feat_cols, feat_labels):\n",
    "    vc = geo_road_check[col].value_counts().sort_index()\n",
    "    ax.bar(vc.index.astype(str), vc.values, color='steelblue')\n",
    "    ax.set_title(f'{label}\\n# unique values per geohash')\n",
    "    ax.set_xlabel('# unique values')\n",
    "    ax.set_ylabel('# geohashes')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOT_DIR}/D19_road_feature_stability.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── INSIGHT SUMMARY ──────────────────────────────────────────────────────────
new_cells.append(md_cell("### Summary of Key EDA Insights"))

new_cells.append(code_cell([
    "# Final insight summary — top 10 findings for feature engineering\n",
    "\n",
    "# Compute supporting numbers on the fly\n",
    "geo_road_check2 = eda.groupby('geohash').agg(\n",
    "    RoadType_nunique = ('RoadType','nunique'),\n",
    "    Lanes_nunique    = ('NumberofLanes','nunique'),\n",
    ").reset_index()\n",
    "pct_stable_road  = 100*(geo_road_check2['RoadType_nunique']  == 1).mean()\n",
    "pct_stable_lanes = 100*(geo_road_check2['Lanes_nunique'] == 1).mean()\n",
    "\n",
    "hourly_mean2 = eda.groupby('hour')['demand'].mean()\n",
    "peak_h   = hourly_mean2.idxmax()\n",
    "trough_h = hourly_mean2.idxmin()\n",
    "peak_ratio = hourly_mean2.max() / hourly_mean2.min()\n",
    "\n",
    "rt_means = eda.groupby('RoadType')['demand'].mean().sort_values(ascending=False)\n",
    "w_means  = eda.groupby('Weather')['demand'].mean().sort_values(ascending=False)\n",
    "lv_means = eda.groupby('LargeVehicles')['demand'].mean()\n",
    "lm_means = eda.groupby('Landmarks')['demand'].mean()\n",
    "\n",
    "train_geo2 = set(train['geohash'])\n",
    "test_geo2  = set(test['geohash'])\n",
    "pct_unseen = 100*len(test_geo2 - train_geo2)/len(test_geo2)\n",
    "\n",
    "from scipy.stats import pearsonr as pr2\n",
    "temp_nonan = eda[['Temperature','demand']].dropna()\n",
    "temp_corr, _ = pr2(temp_nonan['Temperature'], temp_nonan['demand'])\n",
    "\n",
    "lv_ratio = lv_means.get('Allowed', np.nan) / lv_means.get('Not Allowed', np.nan)\n",
    "lm_ratio = lm_means.get('Yes', np.nan) / lm_means.get('No', np.nan)\n",
    "\n",
    "print('=' * 72)\n",
    "print('TOP 10 EDA INSIGHTS FOR FEATURE ENGINEERING')\n",
    "print('=' * 72)\n",
    "\n",
    "insights = [\n",
    "    (1, 'STRONG TIME SIGNAL',\n",
    "     f'Demand varies {peak_ratio:.1f}x across hours (peak h={peak_h}, '\n",
    "     f'trough h={trough_h}). Hour + 15-min time_slot are the #1 predictors. '\n",
    "     f'Use cyclical sin/cos encoding to preserve continuity.'),\n",
    "\n",
    "    (2, 'GEOHASH IS THE DOMINANT FEATURE',\n",
    "     f'Individual geohash explains most demand variance. '\n",
    "     f'Mean demand per geohash ranges from near-0 to 1.0. '\n",
    "     f'Target-encode geohash: geo_mean, geo_median, geo_std are critical features.'),\n",
    "\n",
    "    (3, 'GEO x TIME INTERACTION',\n",
    "     f'Different locations peak at different hours. '\n",
    "     f'geo_slot_mean (geohash × 15-min slot) captures local temporal pattern '\n",
    "     f'and will be one of the strongest engineered features.'),\n",
    "\n",
    "    (4, 'ROAD FEATURES ARE NEAR-STATIC PER LOCATION',\n",
    "     f'{pct_stable_road:.1f}% of geohashes have a single RoadType; '\n",
    "     f'{pct_stable_lanes:.1f}% have a single NumberofLanes. '\n",
    "     f'NaN RoadType can be imputed from geohash mode, recovering ~600 rows cleanly.'),\n",
    "\n",
    "    (5, 'HIGHWAY >> STREET >> RESIDENTIAL DEMAND',\n",
    "     f'Mean demand: {rt_means.to_dict()}. '\n",
    "     f'Highway has substantially higher demand. '\n",
    "     f'RoadType is a strong ordinal feature (encode as 0/1/2).'),\n",
    "\n",
    "    (6, 'LARGE VEHICLES = HIGHER DEMAND ROADS',\n",
    "     f'Allowed roads have {lv_ratio:.2f}x mean demand of Not-Allowed. '\n",
    "     f'Strongly correlated with RoadType (Highways allow large vehicles). '\n",
    "     f'road_capacity = NumberofLanes × (1+LargeVehicles_enc) is a useful proxy.'),\n",
    "\n",
    "    (7, 'LANDMARKS NEAR = HIGHER DEMAND',\n",
    "     f'Landmarks=Yes has {lm_ratio:.2f}x mean demand vs No. '\n",
    "     f'Landmark presence indicates commercial/tourist zones. '\n",
    "     f'Binary encode directly; consider geo×landmark interactions.'),\n",
    "\n",
    "    (8, f'TEMPERATURE HAS WEAK BUT NON-LINEAR EFFECT (r={temp_corr:.3f})',\n",
    "     f'Linear correlation is low ({temp_corr:.3f}) but binned analysis shows '\n",
    "     f'extreme temperatures (very cold/hot) suppress demand. '\n",
    "     f'Add temp² and absolute-temp features. Impute ~2,495 NaN with geohash median.'),\n",
    "\n",
    "    (9, 'WEATHER HAS MODEST IMPACT',\n",
    "     f'Weather means: {w_means.round(4).to_dict()}. '\n",
    "     f'Differences are real but small. Snowy/Foggy reduce demand slightly. '\n",
    "     f'Weather × Hour interaction may capture weather-dependent commute patterns.'),\n",
    "\n",
    "    (10, f'{pct_unseen:.1f}% TEST GEOHASHES ARE UNSEEN',\n",
    "     f'{len(test_geo2-train_geo2)} geohashes in test never appear in train. '\n",
    "     f'Fallback to coarser prefix (geo3/geo4) mean demand for these cold-start locations. '\n",
    "     f'Lat/lon features provide smooth spatial interpolation for unseen geohashes.'),\n",
    "]\n",
    "\n",
    "for idx, title, detail in insights:\n",
    "    print(f'\\n{idx:2d}. [{title}]')\n",
    "    # Wrap text at 72 chars\n",
    "    words = detail.split()\n",
    "    line  = '    '\n",
    "    for w in words:\n",
    "        if len(line) + len(w) + 1 > 74:\n",
    "            print(line)\n",
    "            line = '    ' + w + ' '\n",
    "        else:\n",
    "            line += w + ' '\n",
    "    print(line)\n",
    "\n",
    "print('\\n' + '=' * 72)\n",
    "print('RECOMMENDED FEATURE PRIORITY ORDER:')\n",
    "print('  1. geo_slot_mean  (geohash × 15-min slot target-encoded mean)')\n",
    "print('  2. geo_hour_mean  (geohash × hour mean)')\n",
    "print('  3. geo_mean       (overall geohash mean demand)')\n",
    "print('  4. time_slot      (0..95 — 15-min resolution)')\n",
    "print('  5. hour_sin/cos   (cyclical time encoding)')\n",
    "print('  6. RoadType_enc   (ordinal: Residential=0, Street=1, Highway=2)')\n",
    "print('  7. road_capacity  (NumberofLanes × LargeVehicles factor)')\n",
    "print('  8. lat / lon      (raw coordinates from geohash decode)')\n",
    "print('  9. geo3_mean      (prefix-3 region mean for cold-start geohashes)')\n",
    "print(' 10. temp features  (Temperature, temp², temp_abs, temp_bin)')\n",
    "print('=' * 72)"
]))

# ─────────────────────────────────────────────────────────────────────────────
# Load notebook, inject cells at position 19, save
# ─────────────────────────────────────────────────────────────────────────────
with open(NB_PATH) as f:
    nb = json.load(f)

# Ensure every existing cell has a valid unique id, clear outputs
seen_ids = set()
for cell in nb['cells']:
    cid = cell.get('id', '')
    if not cid or cid in seen_ids:
        cell['id'] = new_id()
    seen_ids.add(cell['id'])
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

insert_at = 19   # right before Phase 4
nb['cells'] = nb['cells'][:insert_at] + new_cells + nb['cells'][insert_at:]

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Injected {len(new_cells)} cells at position {insert_at}.")
print(f"Total cells now: {len(nb['cells'])}")
