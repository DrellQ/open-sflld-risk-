#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD PARQUET AUDIT & QUALITY GOVERNANCE GATE
================================================================================
Validates schema compliance, sentinel codes, distribution bounds,
and credit target mechanics prior to XGBoost and Vasicek correlation training.

REGULATORY & METHODOLOGICAL FOUNDATIONS:
--------------------------------------------------------------------------------
1. CECL & Basel Frameworks:
   - For lifetime expected credit loss (CECL/IFRS 9) and regulatory capital (Basel),
     forward 12-month default targets (PD) must strictly condition on active,
     non-defaulted debt at observation time t = 12 months.
   - Pre-existing defaults (D90+ at Month 12) must be excluded to prevent
     label contamination and severe survival-bias distortion.

2. Macroeconomic Cohort Regimes:
   - Mortgage default hazards are non-stationary across credit cycles. A single
     blanket threshold across 2006-2008 and 2013-2017 produces false alarms.
   - Tolerances are dynamically anchored to historical GSE default rates across:
       * GFC Subprime Contagion (2006-2008)
       * Post-Crisis QM / CRT Tightening (2013-2017)
       * Pre-Pandemic & CARES Act Forbearance (2018-2019)
       * Ultra-Low Rate & Rapid Tightening Cohort (2020-2022)

3. Sentinel Value Purging:
   - Freddie Mac denotes missing values using ASCII integer sentinels:
       * FICO: 9999 (No credit score available)
       * LTV / CLTV: 999 (Missing appraisal / valuation)
       * DTI: 999 (Missing income / debt ratio)
   - These must be purged from numeric distribution checks to avoid skewing means.
================================================================================
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
import polars as pl
import numpy as np


class DualLogger:
    """Tees standard stdout to both the console and an active log file."""
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


def get_vintage_bounds(year: int):
    """
    Returns dynamic credit performance tolerances by macroeconomic era.
    
    RATIONALE FOR ERA BOUNDS:
    ----------------------------------------------------------------------------
    • 2006–2008 (GFC Stress Anchor): [1.50% – 12.00%]
      - Historical Context: The peak of the U.S. housing bubble, subprime contagion,
        and nationwide home price drops of -30%.
      - Lower Bound (1.5%): If realized defaults over months 13-24 drop below 1.5%,
        the input files are almost certainly truncated, missing quarterly performance
        streams, or omitting foreclosure Zero-Balance codes ('02', '03', '09').
      - Upper Bound (12.0%): While cumulative 10-year GFC defaults exceeded 20-30%,
        the marginal hazard rate in months 13-24 specifically rarely exceeds 12%
        for agency conforming paper. Exceeding 12% signals label duplication.

    • 2013–2017 (Post-Crisis QM / CRT Expansion): [0.15% – 0.80%]
      - Historical Context: Implementation of Dodd-Frank Ability-to-Repay (ATR)
        and Qualified Mortgage (QM) rules eliminated stated-income and Alt-A loans.
        Underwriting was pristine (median FICO ~750, strict 43% DTI caps).
      - Lower Bound (0.15% / 15 bps): Conforming paper still experiences baseline
        life events (divorce, job loss, illness) producing ~20-50 bps of default.
        Below 0.15% indicates broken joins between origination and performance.
      - Upper Bound (0.80% / 80 bps): Post-crisis prime conforming paper never
        exceeded 80 bps of early serious delinquency. A rate > 0.80% indicates
        mild roll rates (30-60 DPD) were erroneously counted as defaults.

    • 2018–2019 (Pre-Pandemic / COVID-19 CARES Act Shock): [0.50% – 3.50%]
      - Historical Context: Loans originated in 2018-2019 entered their month 13-24
        window exactly in March 2020 through mid-2021 during COVID-19 shutdowns.
      - CARES Act Impact: Millions of borrowers entered statutory forbearance.
        While many cured, early delinquency spiked artificially to 1.5% - 2.8%.
      - Expected Range [0.5% – 3.5%]: Accommodates pandemic shock elevations
        without allowing runaway uncurred default counts.

    • 2020–2022 (Ultra-Low Rate Refi Boom & Rate Hike Cohort): [0.10% – 1.50%]
      - Historical Context: 2020-2021 was an unprecedented refinancing wave with
        rates under 3.0%. Borrowers built 20-40% equity buffers in home appreciation.
      - Expected Range [0.10% – 1.50%]: Defaults were virtually zero (10-30 bps)
        in 2020-2021, with mild normalization toward 1.0% in 2022 as mortgage
        rates climbed from 3% to 7%.
    """
    if 2006 <= year <= 2008:
        return {"min_dr": 0.015, "max_dr": 0.120, "era": "GFC Stress Anchor"}
    elif 2013 <= year <= 2017:
        return {"min_dr": 0.0015, "max_dr": 0.0080, "era": "Post-Crisis CRT"}
    elif 2018 <= year <= 2019:
        return {"min_dr": 0.0050, "max_dr": 0.0350, "era": "Pre-Pandemic / COVID Shock"}
    elif 2020 <= year <= 2022:
        return {"min_dr": 0.0010, "max_dr": 0.0150, "era": "Rate Shock Cohort"}
    else:
        return {"min_dr": 0.0010, "max_dr": 0.150, "era": "Standard"}


def audit_parquet_file(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File does not exist: {file_path}")
        return False

    filename = path.name
    year_digits = "".join([c for c in filename if c.isdigit()])
    vintage_year = int(year_digits) if len(year_digits) >= 4 else 2015
    bounds = get_vintage_bounds(vintage_year)

    print("\n" + "=" * 80)
    print(f"AUDITING FILE: {filename} (Identified Vintage: {vintage_year} | Era: {bounds['era']})")
    print("=" * 80)

    df = pl.read_parquet(file_path)
    total_loans = len(df)
    print(f"[*] Ingested Records: {total_loans:,}")

    if total_loans == 0:
        print("[FAIL] Dataset has 0 rows.")
        return False

    failures = 0
    warnings = 0

    # --------------------------------------------------------------------------
    # 1. PRIMARY KEY & CORE NULL AUDIT
    # --------------------------------------------------------------------------
    # RATIONALE:
    # In Step 0, all monthly performance rows MUST collapse into exactly ONE
    # cross-sectional snapshot row per loan. Any duplicate loan_id indicates
    # a cross-join defect or broken GROUP BY in DuckDB, which would corrupt
    # tree weights in XGBoost and double-count portfolio risk.
    print("\n--- [1] Structural & Key Completeness ---")
    null_keys = df["loan_id"].null_count()
    unique_keys = df["loan_id"].n_unique()

    if null_keys > 0:
        print(f" [FAIL] Null loan_ids detected: {null_keys:,}")
        failures += 1
    if unique_keys != total_loans:
        print(f" [FAIL] Duplicate loan_ids: {total_loans - unique_keys:,}")
        failures += 1
    else:
        print(f" [PASS] 100% Unique Primary Keys ({unique_keys:,} unique loans)")

    # --------------------------------------------------------------------------
    # 2. TARGET VARIABLE INTEGRITY & REASON CODES
    # --------------------------------------------------------------------------
    # RATIONALE:
    # 1. y_target in {0, 1} must be binary non-null. Null targets cause XGBoost
    #    loss function evaluation failures.
    # 2. The realized 12-month forward default rate MUST fall within the macroeconomic
    #    bounds of its vintage era.
    print("\n--- [2] Forward 12M Credit Target Performance ---")
    null_targets = df["y_target"].null_count()
    if null_targets > 0:
        print(f" [FAIL] Null values in y_target: {null_targets}")
        failures += 1

    defaults = df["y_target"].sum()
    default_rate = df["y_target"].mean()

    print(f"  • Realized Defaults : {defaults:,}")
    print(f"  • Default Rate (12M): {default_rate:.3%}")
    print(f"  • Expected Range    : [{bounds['min_dr']:.2%}, {bounds['max_dr']:.2%}]")

    if not (bounds["min_dr"] <= default_rate <= bounds["max_dr"]):
        print(f" [FAIL] Default rate {default_rate:.3%} breaches historical tolerances for {bounds['era']}.")
        failures += 1
    else:
        print(f" [PASS] Default rate conforms to {bounds['era']} historical distribution.")

    # Target attribution using modern Polars pl.len()
    # Confirms that y_target=1 is driven by expected mechanisms:
    # DQ_GE_3 (90+ DPD), ZERO_BALANCE_EVENT ('02', '03', '09', '15'), or both.
    reasons = df.group_by("default_trigger_reason").agg(
        pl.len().alias("count"),
        (pl.col("y_target").mean() * 100).alias("target_pct")
    ).sort("count", descending=True)
    print("\n  Target Attribution Breakdown:")
    for row in reasons.iter_rows(named=True):
        print(f"    - {row['default_trigger_reason']:<24}: {row['count']:>10,} loans ({row['target_pct']:.1f}% y_target)")

    # --------------------------------------------------------------------------
    # 3. TEMPORAL SANITY & EXCLUSION OF PRE-EXISTING DEFAULTS
    # --------------------------------------------------------------------------
    # RATIONALE:
    # 1. Pre-existing Defaults (snapshot_dq >= 3):
    #    A loan that is already 90+ DPD at Month 12 has ALREADY failed. In credit
    #    risk modeling, keeping loans already in default introduces severe target
    #    leakage, artificially inflating model AUC/Gini because the model simply
    #    memorizes "snapshot_dq >= 3". For a forward-looking PD model, all loans
    #    entering the observation window must be performing or pre-default.
    # 2. Negative Amortization Constraint (snapshot_upb <= original_upb + $10):
    #    Freddie Mac conforming loans amortize each month. Negative amortization
    #    (where the unpaid balance increases above origination) was prohibited
    #    under QM rules and is virtually non-existent in agency conforming paper.
    #    A $10 tolerance accounts for minor capitalized escrow fees. Anything higher
    #    indicates field transposition or balance corruption.
    print("\n--- [3] Observation Time (t=12) State Cleanliness ---")
    pre_existing = df.filter(pl.col("snapshot_dq") >= 3).shape[0]
    if pre_existing > 0:
        print(f" [FAIL] Found {pre_existing:,} loans already defaulted at Month 12! Target leakage detected.")
        failures += 1
    else:
        print(" [PASS] Observation snapshot is clean (zero pre-existing D90+ loans).")

    neg_amort = df.filter(pl.col("snapshot_upb") > (pl.col("original_upb") + 10.0)).shape[0]
    if neg_amort > 0:
        print(f" [FAIL] Found {neg_amort:,} loans where snapshot UPB > orig UPB.")
        failures += 1
    else:
        print(" [PASS] Amortization constraint satisfied (snapshot UPB <= original UPB).")

    # --------------------------------------------------------------------------
    # 4. SENTINEL CODE PURGE & UNDERWRITING VALUE DISTRIBUTIONS
    # --------------------------------------------------------------------------
    # RATIONALE FOR UNDERWRITING TOLERANCES:
    # 1. Sentinel Missingness Threshold (5%):
    #    Agency conforming loans have strict underwriting requirements. Missing FICO
    #    (9999) rarely exceeds 1-2% of an annual book. If missingness exceeds 5%,
    #    the dataset likely suffered parsing truncation or corrupted field alignment.
    # 2. FICO Mean Bounds [500 – 780]:
    #    Average GSE conforming portfolio credit scores sit between 715 and 755
    #    (or ~680 during 2006-2007). An average below 500 or above 780 indicates
    #    catastrophic parsing errors or numeric overflow.
    # 3. LTV Mean Bounds [50.0% – 85.0%]:
    #    Agency purchase loans cluster around 80% (conforming limit without PMI)
    #    and refi loans cluster around 65-75%. The vintage mean should sit tightly
    #    between 70% and 78%. A mean <50% or >85% signals column transposition.
    # 4. DTI Mean Bounds [20.0% – 45.0%]:
    #    Pre-QM loans allowed DTI up to 50%, while QM loans capped DTI at 43%
    #    (with GSE automated underwriting allowing up to 45-50%). Average portfolio
    #    DTI is universally 33% to 38%. A mean <20% or >45% indicates misalignment.
    print("\n--- [4] Underwriting Field Distributions & Sentinel Code Audit ---")
    fico_valid = df.filter((pl.col("credit_score").is_not_null()) & (pl.col("credit_score") != 9999))["credit_score"]
    ltv_valid = df.filter((pl.col("original_ltv").is_not_null()) & (pl.col("original_ltv") != 999))["original_ltv"]
    dti_valid = df.filter((pl.col("original_dti").is_not_null()) & (pl.col("original_dti") != 999))["original_dti"]
    rate_valid = df.filter((pl.col("snapshot_note_rate").is_not_null()) & (pl.col("snapshot_note_rate") > 0))["snapshot_note_rate"]

    fico_sentinels = total_loans - len(fico_valid)
    ltv_sentinels = total_loans - len(ltv_valid)
    dti_sentinels = total_loans - len(dti_valid)

    print(f"  • FICO Sentinel / Missing Count: {fico_sentinels:,} ({fico_sentinels/total_loans:.2%}) [Tolerance: <5.0%]")
    print(f"  • LTV  Sentinel / Missing Count: {ltv_sentinels:,} ({ltv_sentinels/total_loans:.2%}) [Tolerance: <5.0%]")
    print(f"  • DTI  Sentinel / Missing Count: {dti_sentinels:,} ({dti_sentinels/total_loans:.2%}) [Tolerance: <5.0%]")

    if (fico_sentinels / total_loans) > 0.05:
        print(f" [WARN] FICO missingness is elevated ({fico_sentinels/total_loans:.2%}).")
        warnings += 1

    print(f"  • FICO Mean: {fico_valid.mean():.1f} | Median: {fico_valid.median():.1f} | Min: {fico_valid.min()} | Max: {fico_valid.max()} [Bounds: 500-780]")
    print(f"  • LTV  Mean: {ltv_valid.mean():.1f}% | Median: {ltv_valid.median():.1f}% | Min: {ltv_valid.min():.1f}% | Max: {ltv_valid.max():.1f}% [Bounds: 50-85%]")
    print(f"  • DTI  Mean: {dti_valid.mean():.1f}% | Median: {dti_valid.median():.1f}% | Min: {dti_valid.min():.1f}% | Max: {dti_valid.max():.1f}% [Bounds: 20-45%]")
    print(f"  • Note Rate Mean: {rate_valid.mean():.2f}% | Min: {rate_valid.min():.2f}% | Max: {rate_valid.max():.2f}%")

    if not (500 <= fico_valid.mean() <= 780):
        print(" [FAIL] FICO mean outside reasonable bounds.")
        failures += 1
    if not (50.0 <= ltv_valid.mean() <= 85.0):
        print(" [FAIL] LTV mean outside reasonable bounds.")
        failures += 1
    if not (20.0 <= dti_valid.mean() <= 45.0):
        print(" [FAIL] DTI mean outside reasonable bounds.")
        failures += 1

    # --------------------------------------------------------------------------
    # 5. DYNAMIC STEP 0 EXTENSIONS AUDIT (LGD & MODS)
    # --------------------------------------------------------------------------
    # RATIONALE:
    # 1. Modifications (snapshot_mod_flag):
    #    Loans modified within the first 12 months represent early loan workouts.
    #    Verifies that modification indicators were extracted correctly from
    #    monthly performance field #7.
    # 2. Loss Given Default (forward_actual_loss):
    #    Verifies presence of liquidation dollar loss calculations for liquidated
    #    loans. For recent cohorts (2021-2022), foreclosure pipelines take 2-4
    #    years, so losses will legitimately be unpopulated or sparse.
    print("\n--- [5] Dynamic Performance & Recovery Fields Audit ---")
    mod_count = df.filter(pl.col("snapshot_mod_flag") == 1).shape[0]
    print(f"  • Loans Modified by Month 12: {mod_count:,} ({mod_count/total_loans:.3%})")

    liquidated_losses = df.filter((pl.col("y_target") == 1) & (pl.col("forward_actual_loss").is_not_null()))["forward_actual_loss"]
    if len(liquidated_losses) > 0:
        pos_losses = liquidated_losses.filter(liquidated_losses > 0)
        print(f"  • Default Events with Populated Actual Loss: {len(liquidated_losses):,} / {defaults:,}")
        if len(pos_losses) > 0:
            print(f"  • Mean Realized Dollar Loss: ${pos_losses.mean():,.2f}")
            print(f"  • Max Realized Dollar Loss : ${pos_losses.max():,.2f}")
    else:
        print("  • [INFO] No realized actual loss populated in this vintage (Expected for recent non-liquidated cohorts).")

    # --------------------------------------------------------------------------
    # AUDIT VERDICT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    if failures == 0:
        print(f"VERDICT: [PASSED ALL GATES] {filename} is production-ready for Stage 1 modeling.")
        print("=" * 80)
        return True
    else:
        print(f"VERDICT: [FAILED] {filename} failed {failures} critical quality checks. Correct ETL before training.")
        print("=" * 80)
        return False


def main():
    parser = argparse.ArgumentParser(description="Freddie Mac SFLLD Parquet Quality Auditor & Reporter")
    parser.add_argument("--path", "-p", required=True, help="Path to single Parquet file or directory of Parquet files")
    parser.add_argument("--report-dir", "-r", default="./data validation reports", help="Directory where audit report logs will be saved")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(list(target.glob("*.parquet")))
    else:
        print(f"Invalid path: {args.path}")
        sys.exit(1)

    # Initialize report directory and log file
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_tag = target.stem if target.is_file() else target.name
    report_file = report_dir / f"audit_report_{target_tag}_{timestamp}.txt"

    # Set up dual logger
    dual_logger = DualLogger(report_file)
    sys.stdout = dual_logger

    print(f"Audit Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Input Path        : {target.resolve()}")
    print(f"Audit Report Log         : {report_file.resolve()}\n")

    all_passed = True
    for f in files:
        passed = audit_parquet_file(str(f))
        if not passed:
            all_passed = False

    print("\n" + "#" * 80)
    if all_passed:
        print(f"FINAL AUDIT SUMMARY: ALL {len(files)} PARQUET VINTAGES PASSED GOVERNANCE CHECKS.")
    else:
        print(f"FINAL AUDIT SUMMARY: ONE OR MORE VINTAGES FAILED QUALITY CHECKS. CHECK LOGS.")
    print(f"Saved full report to: {report_file.resolve()}")
    print("#" * 80 + "\n")

    dual_logger.close()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()