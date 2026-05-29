import pandas as pd
import numpy as np
import sys
import io

BASE = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/dataset"

train = pd.read_csv(f"{BASE}/train.csv")
test  = pd.read_csv(f"{BASE}/test.csv")

lines = []

def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    lines.append(msg)

log("=" * 70)
log("TRAIN DATASET")
log("=" * 70)
log(f"\nShape: {train.shape}")
log("\n--- dtypes ---")
log(train.dtypes.to_string())
log("\n--- head(10) ---")
log(train.head(10).to_string())
log("\n--- describe() ---")
log(train.describe(include="all").to_string())

log("\n" + "=" * 70)
log("TEST DATASET")
log("=" * 70)
log(f"\nShape: {test.shape}")
log("\n--- dtypes ---")
log(test.dtypes.to_string())
log("\n--- head(10) ---")
log(test.head(10).to_string())
log("\n--- describe() ---")
log(test.describe(include="all").to_string())

log("\n" + "=" * 70)
log("CATEGORICAL UNIQUE VALUES")
log("=" * 70)
for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
    for df_name, df in [("TRAIN", train), ("TEST", test)]:
        if col in df.columns:
            vals = df[col].unique().tolist()
            log(f"\n[{df_name}] {col} ({df[col].nunique()} unique): {vals}")

log("\n" + "=" * 70)
log("NULL COUNTS")
log("=" * 70)
log("\nTRAIN nulls:")
log(train.isnull().sum().to_string())
log("\nTEST nulls:")
log(test.isnull().sum().to_string())

log("\n" + "=" * 70)
log("DEMAND DISTRIBUTION (train only)")
log("=" * 70)
d = train["demand"]
from scipy.stats import skew, kurtosis
log(f"  mean      : {d.mean():.6f}")
log(f"  median    : {d.median():.6f}")
log(f"  std       : {d.std():.6f}")
log(f"  skewness  : {skew(d.dropna()):.6f}")
log(f"  kurtosis  : {kurtosis(d.dropna()):.6f}")
log(f"  min       : {d.min():.6f}")
log(f"  max       : {d.max():.6f}")
for p in [1, 5, 25, 75, 95, 99]:
    log(f"  p{p:<2}       : {np.percentile(d.dropna(), p):.6f}")

summary_path = "/Users/sudarshansudhakar/Downloads/FlipKart Challenge/traffic_prediction/data_summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(lines))

log(f"\nSaved summary to {summary_path}")
