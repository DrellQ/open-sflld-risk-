#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD STEP 1: MONOTONIC XGBOOST TRAINING & PERSISTENT AUDIT
================================================================================
Trains a domain-constrained gradient boosted decision tree on 2013-2017 QM
vintages, evaluates discrimination and decile calibration on 2018-2019 OOT,
and writes full evaluation reports simultaneously to terminal and disk.
"""

import os
import sys
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from feature_engineering import load_dataset_cohort, FEATURE_COLS, MONOTONIC_CONSTRAINTS


CATEGORICAL_COLS = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "property_type",
    "loan_purpose"
]


class DualLogger:
    """Tee logger writing simultaneously to stdout terminal and an audit report file."""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        if not self.log_file.closed:
            self.log_file.close()


def prepare_xgb_matrix(df: pl.DataFrame):
    """Converts Polars DataFrame to Pandas with explicit category dtypes for XGBoost."""
    pdf = df.to_pandas()
    for col in CATEGORICAL_COLS:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype("category")
    return pdf


def build_decile_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Computes credit risk decile performance, event captures, and cumulative gains."""
    eval_df = pl.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob
    }).with_columns(
        pl.col("y_prob").qcut(bins, labels=[f"D{i}" for i in range(bins, 0, -1)]).alias("decile")
    )

    decile_summary = eval_df.group_by("decile").agg([
        pl.len().alias("loan_count"),
        pl.col("y_true").sum().alias("defaults"),
        pl.col("y_true").mean().alias("observed_default_rate"),
        pl.col("y_prob").mean().alias("mean_predicted_score"),
        pl.col("y_prob").min().alias("min_score"),
        pl.col("y_prob").max().alias("max_score")
    ]).sort("mean_predicted_score", descending=True)

    total_defaults = y_true.sum()
    decile_summary = decile_summary.with_columns([
        (pl.col("defaults") / total_defaults).alias("default_capture_share"),
        ((pl.col("defaults") / total_defaults).cum_sum() * 100).alias("cumulative_gains_pct")
    ])

    return decile_summary


def train_model(report_dir: str = "reports"):
    t0 = time.time()
    
    # Initialize report directory and persistent log file
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_path / f"model_training_report_stage1_{timestamp}.txt"
    
    dual_logger = DualLogger(report_file)
    sys.stdout = dual_logger

    try:
        print("=" * 80)
        print("FREDDIE MAC SFLLD: STAGE 1 XGBOOST PRODUCTION TRAINING RUN")
        print("=" * 80)
        print(f"Run Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report File Output  : {report_file.resolve()}\n")

        # 1. Ingest Data Cohorts
        train_files = sorted(glob.glob("parquet_output/Post_Crisis_2013_2017/*.parquet"))
        oot_files = sorted(glob.glob("parquet_output/Out_Of_Time_2018_2019/*.parquet"))
        
        print(f"[*] Ingesting Training Set (2013-2017) from {len(train_files)} files...")
        X_train_pl, y_train_pl = load_dataset_cohort(train_files)
        
        print(f"[*] Ingesting Out-Of-Time Test Set (2018-2019) from {len(oot_files)} files...")
        X_oot_pl, y_oot_pl = load_dataset_cohort(oot_files)

        X_train = prepare_xgb_matrix(X_train_pl)
        y_train = y_train_pl.to_numpy()
        X_oot = prepare_xgb_matrix(X_oot_pl)
        y_oot = y_oot_pl.to_numpy()

        # 2. Build Monotonic Constraints Tuple
        feature_order = list(X_train.columns)
        constraints = tuple(MONOTONIC_CONSTRAINTS.get(col, 0) for col in feature_order)

        # 3. Model Architecture Parameters
        scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 50,
            "monotone_constraints": constraints,
            "enable_categorical": True,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": 8
        }

        print("\n--- Model Hyperparameters & Architecture ---")
        print(f"  • Training Records     : {len(X_train):,}")
        print(f"  • Training Defaults    : {int(y_train.sum()):,} ({y_train.mean():.4%})")
        print(f"  • OOT Records          : {len(X_oot):,}")
        print(f"  • OOT Defaults         : {int(y_oot.sum()):,} ({y_oot.mean():.4%})")
        print(f"  • Feature Count        : {len(feature_order)}")
        print(f"  • Monotone Constraints : {sum(1 for c in constraints if c != 0)} constrained features")
        print(f"  • Scale Pos Weight     : {scale_pos_weight:.2f}")

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        doot = xgb.DMatrix(X_oot, label=y_oot, enable_categorical=True)

        print("\n[*] Commencing Gradient Boosting (Early stopping patience = 30)...")
        evals = [(dtrain, "train"), (doot, "oot")]
        
        evals_result = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=300,
            evals=evals,
            early_stopping_rounds=30,
            evals_result=evals_result,
            verbose_eval=25
        )

        # 4. Predictions & Discrimination Metrics
        preds_train = model.predict(dtrain)
        preds_oot = model.predict(doot)

        train_auc = roc_auc_score(y_train, preds_train)
        oot_auc = roc_auc_score(y_oot, preds_oot)
        train_prauc = average_precision_score(y_train, preds_train)
        oot_prauc = average_precision_score(y_oot, preds_oot)
        oot_brier = brier_score_loss(y_oot, preds_oot)

        print("\n" + "=" * 80)
        print("STAGE 1 MODEL PERFORMANCE & VALIDATION REPORT")
        print("=" * 80)
        print(f"  Best Boosting Iteration : {model.best_iteration}")
        print(f"  Train ROC-AUC           : {train_auc:.4f}  |  Train PR-AUC : {train_prauc:.4f}")
        print(f"  OOT   ROC-AUC           : {oot_auc:.4f}  |  OOT   PR-AUC : {oot_prauc:.4f}")
        print(f"  OOT Brier Loss Score    : {oot_brier:.5f}")
        print("=" * 80)

        # 5. Risk Decile Lift Table on Out-Of-Time
        print("\n[*] Out-Of-Time (2018-2019) Credit Decile Breakdown:")
        decile_table = build_decile_table(y_oot, preds_oot)
        print(decile_table)

        # 6. Feature Importance (Split Gain Attribution)
        importance = model.get_score(importance_type="gain")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n[*] Top Risk Drivers (Gain Attribution):")
        for feat, gain in sorted_importance[:12]:
            constraint_tag = f"[Monotone: {MONOTONIC_CONSTRAINTS.get(feat, 0):+d}]" if feat in MONOTONIC_CONSTRAINTS else "[Unconstrained]"
            print(f"  • {feat:<24} : Gain = {gain:>10.2f}  {constraint_tag}")

        # 7. Persist Serialized Artifacts
        os.makedirs("models", exist_ok=True)
        model_path = "models/xgboost_sflld_stage1.json"
        model.save_model(model_path)
        print(f"\n[✔] Serialized model saved to {model_path}")
        print(f"[✔] Total pipeline execution time: {(time.time() - t0):.1f}s")
        print(f"[✔] Full performance audit log saved to {report_file.resolve()}\n")

    finally:
        dual_logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Freddie Mac SFLLD Stage 1 XGBoost Trainer")
    parser.add_argument("--report-dir", "-r", default="data validation reports", help="Target folder for audit text report")
    args = parser.parse_args()
    
    train_model(report_dir=args.report_dir)