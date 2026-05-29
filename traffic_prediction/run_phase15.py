"""Execute Phase 15 cells sequentially with progress logging."""
import sys, json, time, traceback

LOG = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/phase15_progress.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# ── bootstrap globals ────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import os, copy, numpy as np, pandas as pd
# import torch FIRST so its OpenMP init wins over LightGBM/CatBoost
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# macOS: avoid segfault from OpenMP conflict between torch + LightGBM/CatBoost
os.environ['KMP_DUPLICATE_LIB_OK']    = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS']          = '1'
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb, xgboost as xgb, joblib
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
BASE = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/dataset"
OUT  = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction"

open(LOG, "w").close()          # reset log
log("Phase 15 runner started")

# ── load Phase 15 cells from notebook ───────────────────────────────────────
nb    = json.load(open(f"{OUT}/notebooks/flipkart_traffic_prediction.ipynb"))
cells = [(i,c) for i,c in enumerate(nb["cells"])
         if c["cell_type"]=="code" and "15." in "".join(c["source"])[:10]]
log(f"Found {len(cells)} Phase 15 code cells: {[i for i,_ in cells]}")

g = {k:v for k,v in globals().items()}   # pass all imports into exec context

for cell_idx, (nb_idx, cell) in enumerate(cells, 1):
    src = "".join(cell["source"])
    tag = src.strip().split("\n")[0][:60]
    log(f"--- Cell {cell_idx}/{len(cells)} (nb_idx={nb_idx}): {tag}")
    t0 = time.time()
    try:
        exec(src, g)
        elapsed = time.time() - t0
        log(f"    OK  ({elapsed:.0f}s)")
    except Exception as e:
        log(f"    FAILED: {e}")
        traceback.print_exc()
        log("Aborting.")
        sys.exit(1)

log("=" * 60)
log("Phase 15 complete.")

# report the generated file
import glob
csvs = sorted(glob.glob(f"{OUT}/submissions/*.csv"))
log("Submissions folder contents:")
for p in csvs:
    df = pd.read_csv(p)
    log(f"  {os.path.basename(p):45s}  shape={df.shape}  demand_mean={df['demand'].mean():.5f}")
