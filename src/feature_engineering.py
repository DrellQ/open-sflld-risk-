#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD STEP 1: PRODUCTION FEATURE ENGINEERING PIPELINE
================================================================================
Extracts observation-clean records (t=12), transforms credit and underwriting 
variables, handles sentinels, and prepares training/evaluation feature sets.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple
import polars as pl
import numpy as np


FEATURE_COLS = [
    # Borrower / Credit
    "credit_score",
    "original_dti",
    "num_borrowers",
    "first_time_homebuyer_flag",
    
    # Collateral / Leverage
    "original_ltv",
    "original_cltv",
    "cltv_ltv_spread",
    "num_units",
    "occupancy_status",
    "property_type",
    
    # Loan Structure / Pricing
    "original_upb",
    "original_interest_rate",
    "original_loan_term",
    "loan_purpose",
    "channel",
    
    # Observation Snapshot (t=12)
    "snapshot_upb",
    "snapshot_dq",
    "snapshot_note_rate",
    "upb_paydown_ratio",
    "rate_delta",
    "snapshot_mod_flag"
]

MONOTONIC_CONSTRAINTS = {
    "credit_score": -1,          # Higher score -> strictly lower risk
    "original_dti": 1,           # Higher DTI -> strictly higher risk
    "original_ltv": 1,           # Higher LTV -> strictly higher risk
    "original_cltv": 1,          # Higher CLTV -> strictly higher risk
    "original_interest_rate": 1, # Higher coupon -> strictly higher risk
    "snapshot_dq": 1,            # Delinquency at t=12 -> strictly higher risk
    "upb_paydown_ratio": 1       # Higher remaining balance ratio -> higher risk
}


def clean_and_transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """Applies sentinel purging, feature derivations, and type casting."""
    return df.with_columns([
        # 1. Clean Sentinel Underwriting Codes to Nulls (typed accurately)
        pl.when(pl.col("credit_score").is_in([9999, 0]))
          .then(None)
          .otherwise(pl.col("credit_score"))
          .alias("credit_score"),

        pl.when(pl.col("original_ltv").is_in([999.0, 0.0]))
          .then(None)
          .otherwise(pl.col("original_ltv"))
          .alias("original_ltv"),

        pl.when(pl.col("original_cltv").is_in([999.0, 0.0]))
          .then(None)
          .otherwise(pl.col("original_cltv"))
          .alias("original_cltv"),

        pl.when(pl.col("original_dti").is_in([999.0, 0.0]))
          .then(None)
          .otherwise(pl.col("original_dti"))
          .alias("original_dti"),

        # 2. Categorical Clean-up & Legacy Harmonization
        pl.col("first_time_homebuyer_flag"),
        pl.col("occupancy_status"),
        pl.when(pl.col("channel") == "T")
          .then(pl.lit("B"))
          .otherwise(pl.col("channel"))
          .alias("channel"),
        pl.col("property_type"),
        pl.col("loan_purpose"),
        pl.col("num_borrowers").fill_null(1),
        pl.col("num_units").fill_null(1),

    ]).with_columns([
        # 3. Engineered Risk Drivers
        (pl.col("original_cltv") - pl.col("original_ltv")).alias("cltv_ltv_spread"),
        (pl.col("snapshot_upb") / pl.col("original_upb")).alias("upb_paydown_ratio"),
        (pl.col("snapshot_note_rate") - pl.col("original_interest_rate")).alias("rate_delta")
    ])

def load_dataset_cohort(file_paths: List[str]) -> Tuple[pl.DataFrame, pl.Series]:
    """Loads Parquet cohorts into an aligned Polars DataFrame."""
    print(f"Loading {len(file_paths)} files...")
    dfs = [pl.read_parquet(f) for f in file_paths]
    combined_df = pl.concat(dfs, how="vertical")
    
    transformed_df = clean_and_transform_features(combined_df)
    
    X = transformed_df.select(FEATURE_COLS)
    y = transformed_df.select("y_target").to_series()
    
    return X, y


if __name__ == "__main__":
    train_paths = sorted(glob.glob("parquet_output/Post_Crisis_2013_2017/*.parquet"))
    oot_paths = sorted(glob.glob("parquet_output/Out_Of_Time_2018_2019/*.parquet"))

    print(f"Found {len(train_paths)} training files (2013-2017).")
    print(f"Found {len(oot_paths)} out-of-time test files (2018-2019).")

    X_train, y_train = load_dataset_cohort(train_paths)
    print(f"Train Shape: {X_train.shape} | Default Rate: {y_train.mean():.4%}")

    X_oot, y_oot = load_dataset_cohort(oot_paths)
    print(f"OOT Shape: {X_oot.shape} | Default Rate: {y_oot.mean():.4%}")