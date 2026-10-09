#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD: PROBABILITY CALIBRATION & RISK CURVE GENERATION
================================================================================
Fits a Platt scaling logistic calibrator on the Out-of-Time (2018-2019) cohort,
serializes the calibrator to models/platt_calibrator_stage1.joblib, and exports
calibration reliability diagrams, ROC curves, and score distributions.
================================================================================
"""

import os
import glob
import joblib
import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve, auc

from feature_engineering import FEATURE_COLS, clean_and_transform_features
from stress_test_var import prepare_xgb_matrix


def run_calibration_and_generate_plots(
    model_path: str = "models/xgboost_sflld_stage1.json",
    calibrator_out: str = "models/platt_calibrator_stage1.joblib",
    fig_dir: str = "reports/figures"
):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load Booster Artifact
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run Stage 1 training first.")
    
    print(f"[*] Loading Stage 1 model from {model_path}...")
    model = xgb.Booster()
    model.load_model(model_path)

    # 2. Ingest Out-Of-Time (2018-2019) for Platt Calibration
    oot_files = sorted(glob.glob("parquet_output/Out_Of_Time_2018_2019/*.parquet"))
    print(f"[*] Ingesting OOT dataset from {len(oot_files)} files...")
    dfs = [pl.read_parquet(f) for f in oot_files]
    raw_oot = pl.concat(dfs, how="vertical")
    
    transformed_oot = clean_and_transform_features(raw_oot)
    X_pl = transformed_oot.select(FEATURE_COLS)
    y_oot = transformed_oot.select("y_target").to_series().to_numpy()

    X_mat = prepare_xgb_matrix(X_pl)
    doot = xgb.DMatrix(X_mat, enable_categorical=True)

    print("[*] Generating uncalibrated raw model predictions...")
    raw_scores = model.predict(doot)

    # 3. Fit Platt Scaler (Logistic Regression on Raw Decision Margin/Probability)
    print("[*] Fitting Platt Scaler (univariate logistic calibration)...")
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(raw_scores.reshape(-1, 1), y_oot)

    calibrated_pd = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    raw_brier = brier_score_loss(y_oot, raw_scores)
    cal_brier = brier_score_loss(y_oot, calibrated_pd)
    oot_roc_auc = roc_auc_score(y_oot, calibrated_pd)

    print("\n" + "=" * 70)
    print("CALIBRATION PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Observed Empirical Default Rate : {y_oot.mean():.4%}")
    print(f"  Raw XGBoost Mean Score          : {raw_scores.mean():.4%}")
    print(f"  Calibrated Mean Probability (PD): {calibrated_pd.mean():.4%}")
    print(f"  Pre-Calibration Brier Score     : {raw_brier:.5f}")
    print(f"  Post-Calibration Brier Score    : {cal_brier:.5f}")
    print(f"  Rank-Ordering ROC-AUC           : {oot_roc_auc:.4f}")
    print("=" * 70)

    # 4. Serialize Calibrator
    joblib.dump(calibrator, calibrator_out)
    print(f"\n[✔] Platt Calibrator saved to {calibrator_out}")

    # 5. Generate Visual Diagnostics
    print("[*] Generating publication plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A. Reliability Diagram
    prob_true_raw, prob_pred_raw = calibration_curve(y_oot, raw_scores, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_oot, calibrated_pd, n_bins=10)

    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(prob_pred_raw, prob_true_raw, "s-", color="#1f77b4", label=f"Raw XGB (Brier={raw_brier:.3f})")
    axes[0].plot(prob_pred_cal, prob_true_cal, "o-", color="#ff7f0e", label=f"Platt (Brier={cal_brier:.3f})")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Empirical Default Frequency")
    axes[0].set_title("Reliability Diagram (OOT 2018-2019)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # B. ROC Curve
    fpr, tpr, _ = roc_curve(y_oot, calibrated_pd)
    axes[1].plot(fpr, tpr, color="#2ca02c", lw=2, label=f"OOT ROC (AUC = {oot_roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Out-of-Time Receiver Operating Characteristic")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # C. Score Distribution Shift
    axes[2].hist(raw_scores, bins=50, alpha=0.5, color="#1f77b4", label="Raw Margins", density=True)
    axes[2].hist(calibrated_pd, bins=50, alpha=0.6, color="#ff7f0e", label="Calibrated PD", density=True)
    axes[2].set_xlabel("Predicted Probability / Score")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Score Distribution: Pre vs Post Calibration")
    axes[2].legend(loc="upper right")
    axes[2].set_yscale("log")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    output_path = os.path.join(fig_dir, "model_calibration_and_roc.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✔] Risk curves saved to {output_path}\n")


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    run_calibration_and_generate_plots()