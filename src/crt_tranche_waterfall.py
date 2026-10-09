#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD: CRT TRANCHE WATERFALL & ATTACHMENT/DETACHMENT LOSS ENGINE
================================================================================
Simulates sequential credit loss allocation across synthetic STACR tranches 
(B-2, B-1, M-2, M-1, A-H) using calibrated Monte Carlo credit loss distributions.
Persists execution logs, loss tables, and severity metrics to terminal and disk.
================================================================================
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import polars as pl
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

from feature_engineering import FEATURE_COLS, clean_and_transform_features
from train_xgboost import DualLogger
from stress_test_var import prepare_xgb_matrix


@dataclass
class CRTTranche:
    name: str
    attachment: float  # e.g., 0.005 for 0.5%
    detachment: float  # e.g., 0.015 for 1.5%

    @property
    def thickness(self) -> float:
        return self.detachment - self.attachment


def allocate_losses_to_tranches(
    losses_dollar: np.ndarray, 
    portfolio_upb: float, 
    tranches: List[CRTTranche]
) -> pl.DataFrame:
    """Applies sequential structural loss allocations across tranches."""
    records = []
    
    for tr in tranches:
        tranche_size = tr.thickness * portfolio_upb
        attach_dollar = tr.attachment * portfolio_upb
        detach_dollar = tr.detachment * portfolio_upb
        
        # Vectorized tranche write-down
        tranche_losses = np.clip(losses_dollar - attach_dollar, 0.0, tranche_size)
        
        expected_write_down = float(np.mean(tranche_losses))
        var_95 = float(np.percentile(tranche_losses, 95.0))
        var_99 = float(np.percentile(tranche_losses, 99.0))
        var_999 = float(np.percentile(tranche_losses, 99.9))
        loss_severity_pct = (expected_write_down / tranche_size) * 100.0

        records.append({
            "tranche": tr.name,
            "attachment_pct": tr.attachment * 100.0,
            "detachment_pct": tr.detachment * 100.0,
            "thickness_pct": tr.thickness * 100.0,
            "tranche_size_billions": tranche_size / 1e9,
            "expected_loss_millions": expected_write_down / 1e6,
            "loss_severity_pct": loss_severity_pct,
            "var_95_millions": var_95 / 1e6,
            "var_99_millions": var_99 / 1e6,
            "var_999_millions": var_999 / 1e6
        })

    return pl.DataFrame(records)


def run_tranche_engine(
    report_dir: str = "data validation reports",
    fig_dir: str = "reports/figures",
    n_simulations: int = 20_000
):
    t0 = time.time()
    
    # 1. Initialize report directory and DualLogger
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_path / f"crt_tranche_waterfall_report_{timestamp}.txt"
    
    dual_logger = DualLogger(report_file)
    sys.stdout = dual_logger

    try:
        print("=" * 80)
        print("FREDDIE MAC SFLLD: CRT TRANCHE WATERFALL & ATTACHMENT SIMULATION")
        print("=" * 80)
        print(f"Run Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report File Log  : {report_file.resolve()}\n")

        # 2. Ingest Pre-Trained Artifacts
        model_path = "models/xgboost_sflld_stage1.json"
        calibrator_path = "models/platt_calibrator_stage1.joblib"
        
        if not os.path.exists(model_path) or not os.path.exists(calibrator_path):
            raise FileNotFoundError("Missing Stage 1 model or Platt calibrator. Run Stages 1 & 2 first.")

        print(f"[*] Ingesting model artifact: {model_path}...")
        model = xgb.Booster()
        model.load_model(model_path)
        
        print(f"[*] Ingesting Platt calibrator: {calibrator_path}...")
        calibrator = joblib.load(calibrator_path)

        # 3. Load Modern Portfolio Exposure (2021-2022 Cohort)
        rate_files = sorted(glob.glob("parquet_output/Increasing_Rate_2021_2022/*.parquet"))
        print(f"[*] Ingesting Rate Shock Cohort (2021-2022) from {len(rate_files)} files...")
        dfs = [pl.read_parquet(f) for f in rate_files]
        raw_rate = pl.concat(dfs, how="vertical")
        t_rate = clean_and_transform_features(raw_rate)

        X_mat = prepare_xgb_matrix(t_rate.select(FEATURE_COLS))
        dmat = xgb.DMatrix(X_mat, enable_categorical=True)

        print("[*] Generating calibrated loan-level default probabilities...")
        raw_p = model.predict(dmat)
        calibrated_pd = calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
        upb_array = t_rate["snapshot_upb"].to_numpy()
        total_portfolio_upb = float(np.sum(upb_array))

        # 4. Monte Carlo Loss Simulation
        print(f"[*] Executing Monte Carlo simulation ({n_simulations:,} iterations across {len(upb_array):,} loans)...")
        batch_size = 50_000
        idx = np.random.choice(len(upb_array), size=min(len(upb_array), batch_size), replace=False)
        sample_upb = upb_array[idx]
        sample_pd = calibrated_pd[idx]
        scaling_factor = total_portfolio_upb / np.sum(sample_upb)

        lgd_mean, lgd_std = 0.333149, 0.233959

        sim_losses = np.zeros(n_simulations)
        for i in range(n_simulations):
            draws = np.random.uniform(0.0, 1.0, size=len(sample_upb))
            defaults = (draws < sample_pd).astype(np.float32)
            lgd = np.clip(np.random.normal(lgd_mean, lgd_std, size=len(sample_upb)), 0.05, 0.95)
            sim_losses[i] = np.sum(defaults * sample_upb * lgd) * scaling_factor

        # 5. Define Structured Agency CRT Stack (STACR Reference Design)
        stacr_structure = [
            CRTTranche("Class B-2 (First Loss)", 0.000, 0.005),
            CRTTranche("Class B-1 (Junior Sub)", 0.005, 0.015),
            CRTTranche("Class M-2 (Mezzanine 2)", 0.015, 0.030),
            CRTTranche("Class M-1 (Mezzanine 1)", 0.030, 0.050),
            CRTTranche("Class A-H (Senior GSE)", 0.050, 1.000),
        ]

        waterfall_df = allocate_losses_to_tranches(sim_losses, total_portfolio_upb, stacr_structure)

        # 6. Print Audit Tables
        print("\n" + "=" * 80)
        print("CRT TRANCHE STRUCTURE & SIMULATED LOSS WATERFALL AUDIT")
        print("=" * 80)
        print(f"  Underlying Portfolio UPB : ${total_portfolio_upb:,.2f}")
        print(f"  Simulation Sample Size   : {n_simulations:,} paths")
        print(f"  Downturn LGD Assumption  : μ = {lgd_mean:.4%}, σ = {lgd_std:.4%}")
        print("=" * 80)
        print(waterfall_df)

        print("\n--- Detailed Tranche Impairment Analysis ---")
        for row in waterfall_df.iter_rows(named=True):
            print(f"  • {row['tranche']:<26}: Attach {row['attachment_pct']:>4.1f}% | Detach {row['detachment_pct']:>5.1f}% | "
                  f"Size: ${row['tranche_size_billions']:>7.2f}B | "
                  f"Expected Loss: ${row['expected_loss_millions']:>8.2f}M ({row['loss_severity_pct']:>5.2f}%) | "
                  f"99.0% VaR Loss: ${row['var_99_millions']:>8.2f}M")

        # 7. Generate and Export Visualization
        tranche_names = waterfall_df["tranche"].to_list()
        expected_losses = (waterfall_df["expected_loss_millions"] / 1e3).to_list()
        var_99_losses = (waterfall_df["var_99_millions"] / 1e3).to_list()

        x = np.arange(len(tranche_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, expected_losses, width, label="Expected Loss ($B)", color="#1f77b4")
        ax.bar(x + width/2, var_99_losses, width, label="99.0% VaR Loss ($B)", color="#d62728")

        ax.set_ylabel("Dollar Loss Incurred ($ Billions)", fontsize=11)
        ax.set_title("CRT / STACR Synthetic Tranche Loss Allocation (2021-2022 Portfolio)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tranche_names, rotation=15, ha="right", fontsize=10)
        ax.legend(framealpha=0.95)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        fig_out = os.path.join(fig_dir, "crt_tranche_loss_distribution.png")
        plt.tight_layout()
        plt.savefig(fig_out, dpi=300)
        plt.close()

        print(f"\n[✔] Visualization saved to: {fig_out}")
        print(f"[✔] Total execution time: {(time.time() - t0):.1f}s")
        print(f"[✔] Persistent audit report written to: {report_file.resolve()}\n")

    finally:
        dual_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freddie Mac SFLLD CRT Tranche Loss Engine")
    parser.add_argument("--report-dir", default="data validation reports", help="Target folder for audit text report")
    parser.add_argument("--fig-dir", default="reports/figures", help="Target folder for generated figures")
    parser.add_argument("--simulations", type=int, default=20_000, help="Number of Monte Carlo paths")
    args = parser.parse_args()

    run_tranche_engine(
        report_dir=args.report_dir,
        fig_dir=args.fig_dir,
        n_simulations=args.simulations
    )