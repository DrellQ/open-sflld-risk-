#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD PHASE 2: MACRO STRESS TESTING & MONTE CARLO VaR PIPELINE
================================================================================
Evaluates Stage 1 XGBoost model against GFC (2006-2008) and Rate Shock (2021-2022),
computes empirical Downturn LGD, simulates 99.0% & 99.9% Portfolio VaR, and writes
all audit metrics to both terminal and persistent report text files.
================================================================================
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import polars as pl
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from feature_engineering import FEATURE_COLS, clean_and_transform_features
from train_xgboost import DualLogger, build_decile_table


CATEGORICAL_DOMAINS = {
    "first_time_homebuyer_flag": ["Y", "N", "U"],
    "occupancy_status": ["P", "S", "I", "U"],
    "channel": ["R", "B", "C", "U"],
    "property_type": ["SF", "CO", "CP", "MH", "PU", "OT"],
    "loan_purpose": ["P", "C", "N", "U"]
}

TRAINED_CATEGORIES = {
    "first_time_homebuyer_flag": ["N", "Y"],
    "occupancy_status": ["I", "P", "S"],
    "channel": ["B", "C", "R"],
    "property_type": ["CO", "CP", "MH", "PU", "SF"],
    "loan_purpose": ["C", "N", "P"],
}


def prepare_xgb_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Converts Polars DataFrame to Pandas with strict category dtypes aligned to training."""
    pdf = df.to_pandas()

    if "channel" in pdf.columns:
        pdf["channel"] = pdf["channel"].replace({"T": "B"})

    for col, categories in TRAINED_CATEGORIES.items():
        if col in pdf.columns:
            # Route any unobserved codes or placeholders to NaN for default split branch
            pdf[col] = pdf[col].apply(lambda x: x if x in categories else np.nan)
            pdf[col] = pd.Categorical(pdf[col], categories=categories)

    return pdf


def evaluate_cohort(model: xgb.Booster, cohort_name: str, file_paths: list) -> Tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Scores an external macroeconomic cohort and returns metrics and underlying dataset."""
    print(f"[*] Ingesting {cohort_name} from {len(file_paths)} files...")
    dfs = [pl.read_parquet(f) for f in file_paths]
    raw_df = pl.concat(dfs, how="vertical")
    
    transformed_df = clean_and_transform_features(raw_df)
    
    X_pl = transformed_df.select(FEATURE_COLS)
    y_true = transformed_df.select("y_target").to_series().to_numpy()
    
    X_mat = prepare_xgb_matrix(X_pl)
    dmat = xgb.DMatrix(X_mat, enable_categorical=True)
    
    y_prob = model.predict(dmat)
    
    auc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    print(f"\n--- Model Performance on {cohort_name} ---")
    print(f"  • Total Loan Records   : {len(y_true):,}")
    print(f"  • Realized Defaults    : {int(y_true.sum()):,} ({y_true.mean():.4%})")
    print(f"  • Mean Predicted Score : {y_prob.mean():.4%}")
    print(f"  • Discrimination ROC   : {auc:.4f}  |  PR-AUC: {prauc:.4f}")
    print(f"  • Brier Calibration    : {brier:.5f}")
    
    decile_table = build_decile_table(y_true, y_prob)
    print(f"\n{cohort_name} Decile Risk Distribution:")
    print(decile_table)
    
    return y_true, y_prob, raw_df


def run_portfolio_var_simulation(
    upb_array: np.ndarray, 
    pd_array: np.ndarray, 
    lgd_mean: float, 
    lgd_std: float, 
    n_simulations: int = 10_000
) -> Dict[str, float]:
    """Vectorized Monte Carlo simulation of portfolio credit loss distribution."""
    n_loans = len(upb_array)
    total_portfolio_upb = float(np.sum(upb_array))
    
    print(f"\n[*] Commencing Monte Carlo VaR Engine ({n_simulations:,} paths across {n_loans:,} loans)...")
    
    batch_size = 50_000
    idx = np.random.choice(n_loans, size=min(n_loans, batch_size), replace=False)
    sample_upb = upb_array[idx]
    sample_pd = pd_array[idx]
    scaling_factor = total_portfolio_upb / np.sum(sample_upb)
    
    sim_losses = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Bernoulli default event draw
        random_draws = np.random.uniform(0.0, 1.0, size=len(sample_upb))
        defaults = (random_draws < sample_pd).astype(np.float32)
        
        # Stochastic truncated LGD draw
        lgd_draws = np.clip(np.random.normal(lgd_mean, lgd_std, size=len(sample_upb)), 0.05, 0.95)
        
        sim_losses[i] = np.sum(defaults * sample_upb * lgd_draws) * scaling_factor
        
    expected_loss = float(np.mean(sim_losses))
    var_95 = float(np.percentile(sim_losses, 95.0))
    var_99 = float(np.percentile(sim_losses, 99.0))
    var_999 = float(np.percentile(sim_losses, 99.9))
    es_99 = float(np.mean(sim_losses[sim_losses >= var_99]))
    
    return {
        "portfolio_upb": total_portfolio_upb,
        "expected_loss": expected_loss,
        "el_rate": expected_loss / total_portfolio_upb,
        "var_95": var_95,
        "var_99": var_99,
        "var_999": var_999,
        "es_99": es_99,
        "unexpected_loss_99": var_99 - expected_loss
    }


def execute_stress_and_var_pipeline(
    report_dir: str = "data validation reports", 
    mode: str = "all", 
    n_simulations: int = 10_000
):
    t0 = time.time()
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_path / f"stress_test_var_report_{timestamp}.txt"
    
    dual_logger = DualLogger(report_file)
    sys.stdout = dual_logger

    try:
        print("=" * 80)
        print("FREDDIE MAC SFLLD: PHASE 2 STRESS TESTING & VALUE-AT-RISK AUDIT")
        print("=" * 80)
        print(f"Run Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Execution Mode   : {mode.upper()}")
        print(f"Report File Log  : {report_file.resolve()}\n")
        
        model_path = "models/xgboost_sflld_stage1.json"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Train Stage 1 first.")
            
        print(f"[*] Ingesting serialized model artifact: {model_path}...")
        model = xgb.Booster()
        model.load_model(model_path)
        
        # 1. GFC Anchor Stress & Empirical LGD Calibration
        lgd_mean, lgd_std = 0.35, 0.15
        if mode in ["all", "stress-only"]:
            gfc_files = sorted(glob.glob("parquet_output/GFC_2006_2008/*.parquet"))
            y_gfc, p_gfc, df_gfc = evaluate_cohort(model, "GFC Stress Anchor (2006-2008)", gfc_files)
            
            loss_records = df_gfc.filter(pl.col("forward_actual_loss").is_not_null() & (pl.col("snapshot_upb") > 0))
            if len(loss_records) > 0:
                lgd_ratios = (loss_records["forward_actual_loss"] / loss_records["snapshot_upb"]).to_numpy()
                valid_lgd = lgd_ratios[(lgd_ratios > 0.0) & (lgd_ratios <= 1.0)]
                if len(valid_lgd) > 0:
                    lgd_mean = float(np.mean(valid_lgd))
                    lgd_std = float(np.std(valid_lgd))
            
            print("\n--- Empirical Downturn LGD Calibration (GFC Anchor) ---")
            print(f"  • Realized Liquidation Events : {len(loss_records):,}")
            print(f"  • Empirical Downturn LGD Mean : {lgd_mean:.4%}")
            print(f"  • Empirical Downturn LGD Std  : {lgd_std:.4%}")

        # 2. Rate Shock Stress Testing & Portfolio VaR
        if mode in ["all", "var-only"]:
            rate_files = sorted(glob.glob("parquet_output/Increasing_Rate_2021_2022/*.parquet"))
            y_rate, p_rate, df_rate = evaluate_cohort(model, "Rate Shock Cohort (2021-2022)", rate_files)
            
            upb_rate = df_rate["snapshot_upb"].to_numpy()

            # --- Calibrated PD Injection ---
            calibrator_path = "models/platt_calibrator_stage1.joblib"
            if os.path.exists(calibrator_path):
                print(f"[*] Ingesting Platt calibrator from {calibrator_path}...")
                calibrator = joblib.load(calibrator_path)
                p_rate_sim = calibrator.predict_proba(p_rate.reshape(-1, 1))[:, 1]
            else:
                p_rate_sim = p_rate

            var_results = run_portfolio_var_simulation(
                upb_rate, p_rate_sim, lgd_mean, lgd_std, n_simulations=n_simulations
            )
            
            print("\n" + "=" * 80)
            print("PORTFOLIO VALUE-AT-RISK & ECONOMIC CAPITAL METRICS (2021-2022 Cohort)")
            print("=" * 80)
            print(f"  Total Portfolio Exposure (UPB): ${var_results['portfolio_upb']:,.2f}")
            print(f"  Expected 12M Dollar Loss (EL) : ${var_results['expected_loss']:,.2f} ({var_results['el_rate']:.4%})")
            print(f"  Value-at-Risk (95.0% VaR)     : ${var_results['var_95']:,.2f}")
            print(f"  Value-at-Risk (99.0% VaR)     : ${var_results['var_99']:,.2f}")
            print(f"  Value-at-Risk (99.9% VaR)     : ${var_results['var_999']:,.2f}")
            print(f"  Expected Shortfall (99.0% ES) : ${var_results['es_99']:,.2f}")
            print(f"  Unexpected Loss / Capital Req : ${var_results['unexpected_loss_99']:,.2f}")
            print("=" * 80)

        print(f"\n[✔] Execution complete in {(time.time() - t0):.1f}s")
        print(f"[✔] Persistent audit report written to: {report_file.resolve()}\n")

    finally:
        dual_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freddie Mac SFLLD Phase 2: Stress Testing & VaR Engine")
    parser.add_argument("--mode", choices=["all", "stress-only", "var-only"], default="all", help="Execution mode")
    parser.add_argument("--report-dir", default="data validation reports", help="Target folder for audit text report")
    parser.add_argument("--simulations", type=int, default=10_000, help="Number of Monte Carlo paths")
    args = parser.parse_args()

    execute_stress_and_var_pipeline(
        report_dir=args.report_dir, 
        mode=args.mode, 
        n_simulations=args.simulations
    )