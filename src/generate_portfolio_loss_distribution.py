#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD: MONTE CARLO VaR LOSS DISTRIBUTION VISUALIZER
================================================================================
Generates high-resolution publication figures of the calibrated credit loss 
distribution with EL, VaR(95/99/99.9), and Expected Shortfall overlays.
================================================================================
"""

import os
import glob
import joblib
import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt

from feature_engineering import FEATURE_COLS, clean_and_transform_features
from stress_test_var import prepare_xgb_matrix, run_portfolio_var_simulation

os.makedirs("reports/figures", exist_ok=True)

# 1. Load Model, Calibrator, and Data
print("[*] Ingesting assets for loss density generation...")
model = xgb.Booster()
model.load_model("models/xgboost_sflld_stage1.json")
calibrator = joblib.load("models/platt_calibrator_stage1.joblib")

rate_files = sorted(glob.glob("parquet_output/Increasing_Rate_2021_2022/*.parquet"))
dfs = [pl.read_parquet(f) for f in rate_files]
raw_rate = pl.concat(dfs, how="vertical")
t_rate = clean_and_transform_features(raw_rate)

X_mat = prepare_xgb_matrix(t_rate.select(FEATURE_COLS))
dmat = xgb.DMatrix(X_mat, enable_categorical=True)

raw_p = model.predict(dmat)
calibrated_pd = calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
upb_array = t_rate["snapshot_upb"].to_numpy()

# 2. Simulate 25,000 paths for a smooth density curve
print("[*] Simulating 25,000 loss iterations...")
n_simulations = 25_000
batch_size = 50_000
idx = np.random.choice(len(upb_array), size=min(len(upb_array), batch_size), replace=False)
sample_upb = upb_array[idx]
sample_pd = calibrated_pd[idx]
scaling_factor = np.sum(upb_array) / np.sum(sample_upb)

lgd_mean, lgd_std = 0.3331, 0.2340

sim_losses_b = np.zeros(n_simulations)
for i in range(n_simulations):
    draws = np.random.uniform(0.0, 1.0, size=len(sample_upb))
    defaults = (draws < sample_pd).astype(np.float32)
    lgd = np.clip(np.random.normal(lgd_mean, lgd_std, size=len(sample_upb)), 0.05, 0.95)
    sim_losses_b[i] = (np.sum(defaults * sample_upb * lgd) * scaling_factor) / 1e9

# Metrics in Billions
el = np.mean(sim_losses_b)
var_95 = np.percentile(sim_losses_b, 95.0)
var_99 = np.percentile(sim_losses_b, 99.0)
var_999 = np.percentile(sim_losses_b, 99.9)
es_99 = np.mean(sim_losses_b[sim_losses_b >= var_99])

# 3. Plotting Publication Figure
fig, ax = plt.subplots(figsize=(12, 6))

counts, bins, patches = ax.hist(
    sim_losses_b, bins=80, density=True, alpha=0.65, color="#1f77b4", edgecolor="#154b73"
)

# Highlight Tail Risk Region (>= 99.0% VaR)
tail_mask = bins[:-1] >= var_99
for patch, is_tail in zip(patches, tail_mask):
    if is_tail:
        patch.set_facecolor("#d62728")
        patch.set_alpha(0.85)

# Vertical Threshold Lines
ax.axvline(el, color="darkgreen", linestyle="--", lw=2, label=f"Expected Loss (EL): ${el:.2f}B")
ax.axvline(var_95, color="#ff7f0e", linestyle="-.", lw=1.8, label=f"95.0% VaR: ${var_95:.2f}B")
ax.axvline(var_99, color="#d62728", linestyle="-", lw=2.2, label=f"99.0% VaR: ${var_99:.2f}B")
ax.axvline(es_99, color="purple", linestyle=":", lw=2.2, label=f"99.0% Expected Shortfall (ES): ${es_99:.2f}B")
ax.axvline(var_999, color="black", linestyle="-", lw=1.5, label=f"99.9% VaR: ${var_999:.2f}B")

# Formatting
ax.set_title("Calibrated Monte Carlo Portfolio Credit Loss Distribution (2021-2022 Cohort)", fontsize=13, fontweight="bold")
ax.set_xlabel("12-Month Realized Credit Losses ($ Billions)", fontsize=11)
ax.set_ylabel("Probability Density", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

fig_path = "reports/figures/portfolio_var_loss_distribution.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[✔] High-resolution loss density saved to {fig_path}")