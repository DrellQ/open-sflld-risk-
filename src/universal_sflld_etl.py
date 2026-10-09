#!/usr/bin/env python3
"""
================================================================================
FREDDIE MAC SFLLD STEP 0: IN-MEMORY ETL, REDUCTION & PARQUET STORAGE
================================================================================
Universal In-Memory Streaming Pipeline for Freddie Mac SFLLD:
- Handles both legacy 26-column (2006-2008) and modern 32-column (2013+) layouts
- Extracts snapshot state at loan age t = 12 months (UPB, DQ, note rate, mod flag, deferred UPB)
- Slices 12-month forward performance outcome window (loan age 13 to 24)
- Employs zero-disk extraction (streams directly from nested ZIPs to memory)
- Compresses to typed, single-precision Parquet with ZSTD compression
================================================================================
"""

import io
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List
import zipfile

import duckdb
import pyarrow as pa
import pyarrow.csv as pa_csv

# Configure structured execution logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Universal_SFLLD")

# ==============================================================================
# SCHEMA DEFINITIONS (Freddie Mac Single-Family Loan-Level Dataset)
# ==============================================================================
# Origination file (orig_*.txt): Static loan & borrower underwriting attributes
ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer_flag",
    "maturity_date", "msa", "mi_percent", "num_units", "occupancy_status",
    "original_cltv", "original_dti", "original_upb", "original_ltv",
    "original_interest_rate", "channel", "ppm_flag", "amortization_type",
    "property_state", "property_type", "postal_code", "loan_id",
    "loan_purpose", "original_loan_term", "num_borrowers", "seller_name",
    "servicer_name", "super_conforming_flag", "pre_harp_loan_id",
    "program_indicator", "harp_indicator", "property_valuation_method",
    "interest_only_indicator", "mi_cancellation_indicator"
]

# Monthly Performance file (perf_*.txt): Dynamic monthly loan tracking
PERF_COLS = [
    "loan_id", "monthly_reporting_period", "current_upb", "delinquency_status",
    "loan_age", "remaining_months", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate",
    "current_deferred_upb", "due_date_last_paid", "mi_recoveries",
    "net_sales_proceeds", "non_mi_recoveries", "expenses", "legal_costs",
    "maintenance_costs", "taxes_insurance", "misc_expenses",
    "actual_loss_calculation", "modification_cost", "step_modification_flag",
    "deferred_payment_plan", "eltv", "zb_removal_upb", "delinquent_accrued_interest",
    "disaster_delinquency_flag", "borrower_assistance_code", "current_month_mod_cost",
    "interest_bearing_upb"
]


def read_table_from_bytes(raw_bytes: bytes, schema_type: str) -> pa.Table:
    """
    Reads raw byte buffers into PyArrow Tables with dynamic column projection.
    Dynamically counts delimiter pipes on line 1 to support both legacy (26-column)
    and modern (32-column) monthly performance layouts.
    """
    first_line = raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    actual_cols_count = len(first_line.split("|"))

    all_cols = ORIG_COLS if schema_type == "orig" else PERF_COLS
    selected_cols = all_cols[:actual_cols_count]
    if actual_cols_count > len(selected_cols):
        for i in range(len(selected_cols), actual_cols_count):
            selected_cols.append(f"extra_col_{i}")

    parse_options = pa_csv.ParseOptions(delimiter="|")
    read_options = pa_csv.ReadOptions(use_threads=True, column_names=selected_cols)

    if schema_type == "perf":
        target_perf = [
            "loan_id",
            "monthly_reporting_period",
            "current_upb",
            "delinquency_status",
            "loan_age",
            "zero_balance_code",
            "current_interest_rate",
            "modification_flag",
            "current_deferred_upb",
            "actual_loss_calculation",
        ]
        include = [c for c in target_perf if c in selected_cols]
        convert_options = pa_csv.ConvertOptions(
            include_columns=include, strings_can_be_null=True
        )
    else:
        convert_options = pa_csv.ConvertOptions(strings_can_be_null=True)

    return pa_csv.read_csv(
        io.BytesIO(raw_bytes),
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )


def process_single_vintage(
    zip_path: str,
    output_parquet: str,
    threads: int = 8,
    zstd_level: int = 3,
):
    """
    Streams nested quarterly archives, aggregates snapshot features & targets
    at Month 12, and outputs an institutional-grade Parquet table.
    """
    t0 = time.time()
    vintage_name = os.path.basename(output_parquet)
    logger.info(f"==> Processing archive: {os.path.basename(zip_path)} -> {vintage_name}")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={threads};")
    con.execute("PRAGMA memory_limit='16GB';")

    # Table schema: expanded with dynamic features and LGD tracking
    con.execute("""
        CREATE TABLE vintage_dataset (
            loan_id VARCHAR, credit_score INT32, first_payment_date INT32, first_time_homebuyer_flag VARCHAR,
            maturity_date INT32, msa INT32, mi_percent FLOAT, num_units INT16, occupancy_status VARCHAR,
            original_cltv FLOAT, original_dti FLOAT, original_upb FLOAT, original_ltv FLOAT,
            original_interest_rate FLOAT, channel VARCHAR, ppm_flag VARCHAR, amortization_type VARCHAR,
            property_state VARCHAR, property_type VARCHAR, postal_code VARCHAR, loan_purpose VARCHAR,
            original_loan_term INT32, num_borrowers INT16, seller_name VARCHAR, servicer_name VARCHAR,
            super_conforming_flag VARCHAR, 
            snapshot_upb FLOAT, snapshot_dq INT32, snapshot_note_rate FLOAT, 
            snapshot_mod_flag INT8, snapshot_deferred_upb FLOAT,
            y_target INT32, default_trigger_reason VARCHAR, forward_actual_loss FLOAT
        );
    """)

    quarters_processed = 0

    with zipfile.ZipFile(zip_path, "r") as outer_zf:
        members = outer_zf.namelist()
        quarterly_zips = [
            m for m in members if m.endswith(".zip") and not m.startswith("__MACOSX/")
        ]
        quarterly_zips.sort()

        if quarterly_zips:
            for q_zip_name in quarterly_zips:
                m_tag = re.search(r"(\d{4}Q[1-4]|Q[1-4]\d{4})", q_zip_name, re.IGNORECASE)
                q_label = m_tag.group(1).upper() if m_tag else os.path.basename(q_zip_name)
                logger.info(f"  Streaming {q_label} directly from in-memory ZIP buffer...")

                nested_bytes = outer_zf.read(q_zip_name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner_zf:
                    inner_names = [
                        n for n in inner_zf.namelist() if not n.startswith("__MACOSX/")
                    ]

                    orig_name = next(
                        (n for n in inner_names if "orig" in n.lower() and n.endswith(".txt")),
                        None,
                    )
                    perf_name = next(
                        (n for n in inner_names if "perf" in n.lower() and n.endswith(".txt")),
                        None,
                    )

                    if not orig_name or not perf_name:
                        txt_files = [n for n in inner_names if n.endswith(".txt")]
                        if len(txt_files) == 2:
                            txt_files.sort(key=lambda x: inner_zf.getinfo(x).file_size)
                            orig_name = txt_files[0]
                            perf_name = txt_files[1]

                    if not orig_name or not perf_name:
                        logger.error(f"  Could not identify orig/perf files in {q_zip_name}")
                        continue

                    orig_bytes = inner_zf.read(orig_name)
                    perf_bytes = inner_zf.read(perf_name)

                    orig_table = read_table_from_bytes(orig_bytes, "orig")
                    perf_table = read_table_from_bytes(perf_bytes, "perf")
                    del orig_bytes, perf_bytes

                    con.register("orig_arrow", orig_table)
                    con.register("perf_arrow", perf_table)

                    # Dynamic inspection of columns for legacy vs modern layouts
                    perf_cols_present = perf_table.column_names

                    rate_col = (
                        "CAST(TRY_CAST(current_interest_rate AS DOUBLE) AS FLOAT)"
                        if "current_interest_rate" in perf_cols_present
                        else "NULL"
                    )
                    mod_col = (
                        "CASE WHEN modification_flag = 'Y' THEN 1 ELSE 0 END"
                        if "modification_flag" in perf_cols_present
                        else "0"
                    )
                    def_upb_col = (
                        "CAST(TRY_CAST(current_deferred_upb AS DOUBLE) AS FLOAT)"
                        if "current_deferred_upb" in perf_cols_present
                        else "0.0"
                    )
                    loss_col = (
                        "CAST(TRY_CAST(actual_loss_calculation AS DOUBLE) AS FLOAT)"
                        if "actual_loss_calculation" in perf_cols_present
                        else "NULL"
                    )

                    con.execute(f"""
                        INSERT INTO vintage_dataset
                        WITH snap_t12 AS (
                            SELECT 
                                loan_id, 
                                CAST(current_upb AS FLOAT) AS snapshot_upb,
                                CASE 
                                    WHEN delinquency_status = 'RA' THEN 99
                                    WHEN delinquency_status IN ('XX', '') THEN NULL
                                    ELSE TRY_CAST(delinquency_status AS INT32)
                                END AS snapshot_dq,
                                {rate_col} AS snapshot_note_rate,
                                {mod_col} AS snapshot_mod_flag,
                                COALESCE({def_upb_col}, 0.0) AS snapshot_deferred_upb
                            FROM perf_arrow 
                            WHERE TRY_CAST(loan_age AS INT32) = 12
                            QUALIFY ROW_NUMBER() OVER(PARTITION BY loan_id ORDER BY monthly_reporting_period DESC) = 1
                        ),
                        target_window AS (
                            SELECT 
                                loan_id,
                                MAX(CASE 
                                    WHEN (TRY_CAST(delinquency_status AS INT32) >= 3 OR delinquency_status = 'RA')
                                      OR zero_balance_code IN ('02', '03', '09', '15')
                                    THEN 1 ELSE 0 
                                END) AS y_target,
                                MAX(CASE WHEN (TRY_CAST(delinquency_status AS INT32) >= 3 OR delinquency_status = 'RA') THEN 1 ELSE 0 END) AS dq90_flag,
                                MAX(CASE WHEN zero_balance_code IN ('02', '03', '09', '15') THEN 1 ELSE 0 END) AS zb_flag,
                                MAX({loss_col}) AS forward_actual_loss
                            FROM perf_arrow 
                            WHERE TRY_CAST(loan_age AS INT32) BETWEEN 13 AND 24
                            GROUP BY loan_id
                        )
                        SELECT 
                            o.loan_id, 
                            TRY_CAST(o.credit_score AS INT32) AS credit_score,
                            TRY_CAST(o.first_payment_date AS INT32) AS first_payment_date, 
                            o.first_time_homebuyer_flag,
                            TRY_CAST(o.maturity_date AS INT32) AS maturity_date, 
                            TRY_CAST(o.msa AS INT32) AS msa,
                            CAST(TRY_CAST(o.mi_percent AS DOUBLE) AS FLOAT) AS mi_percent, 
                            TRY_CAST(o.num_units AS INT16) AS num_units,
                            o.occupancy_status, 
                            CAST(TRY_CAST(o.original_cltv AS DOUBLE) AS FLOAT) AS original_cltv,
                            CAST(TRY_CAST(o.original_dti AS DOUBLE) AS FLOAT) AS original_dti,
                            CAST(TRY_CAST(o.original_upb AS DOUBLE) AS FLOAT) AS original_upb,
                            CAST(TRY_CAST(o.original_ltv AS DOUBLE) AS FLOAT) AS original_ltv,
                            CAST(TRY_CAST(o.original_interest_rate AS DOUBLE) AS FLOAT) AS original_interest_rate,
                            o.channel, o.ppm_flag, o.amortization_type, o.property_state, o.property_type, o.postal_code,
                            o.loan_purpose, 
                            TRY_CAST(o.original_loan_term AS INT32) AS original_loan_term,
                            TRY_CAST(o.num_borrowers AS INT16) AS num_borrowers, 
                            o.seller_name, o.servicer_name,
                            CASE WHEN 'super_conforming_flag' IN (SELECT column_name FROM (DESCRIBE orig_arrow)) THEN o.super_conforming_flag ELSE NULL END AS super_conforming_flag,
                            s.snapshot_upb, 
                            COALESCE(s.snapshot_dq, 0) AS snapshot_dq,
                            COALESCE(s.snapshot_note_rate, CAST(TRY_CAST(o.original_interest_rate AS DOUBLE) AS FLOAT)) AS snapshot_note_rate,
                            COALESCE(s.snapshot_mod_flag, 0) AS snapshot_mod_flag,
                            COALESCE(s.snapshot_deferred_upb, 0.0) AS snapshot_deferred_upb,
                            COALESCE(t.y_target, 0) AS y_target,
                            CASE 
                                WHEN t.dq90_flag = 1 AND t.zb_flag = 1 THEN 'DQ_AND_ZERO_BALANCE'
                                WHEN t.dq90_flag = 1 THEN 'DQ_GE_3'
                                WHEN t.zb_flag = 1 THEN 'ZERO_BALANCE_EVENT'
                                ELSE 'PERFORMING_OR_PREPAID'
                            END AS default_trigger_reason,
                            t.forward_actual_loss
                        FROM orig_arrow o
                        INNER JOIN snap_t12 s ON o.loan_id = s.loan_id
                        LEFT JOIN target_window t ON o.loan_id = t.loan_id
                        WHERE s.snapshot_upb > 0
                        AND (s.snapshot_dq < 3 OR s.snapshot_dq IS NULL)
                        AND s.snapshot_upb <= (CAST(TRY_CAST(o.original_upb AS DOUBLE) AS FLOAT) + 10.0);
                    """)

                    con.unregister("orig_arrow")
                    con.unregister("perf_arrow")
                    del orig_table, perf_table
                    quarters_processed += 1

    if quarters_processed == 0:
        logger.error(f"Could not process any quarters in {zip_path}")
        con.close()
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    con.execute(
        f"COPY vintage_dataset TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION 'ZSTD', COMPRESSION_LEVEL {zstd_level});"
    )
    summary = con.execute(
        "SELECT COUNT(*), SUM(y_target), AVG(y_target)*100 FROM vintage_dataset;"
    ).fetchone()
    con.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    logger.info(
        f"  [SUCCESS] {summary[0]:,} loans collapsed across {quarters_processed} quarters | "
        f"Defaults: {summary[1]:,} ({summary[2]:.2f}%) | "
        f"Parquet: {size_mb:.2f} MB in {elapsed:.1f}s\n"
    )


def parse_vintage_groups(group_tokens: List[str]) -> List[int]:
    """Expands range tokens like '2006-2008' or '2013-2017' into individual year integers."""
    years = []
    for token in group_tokens:
        if "-" in token:
            start, end = token.split("-")
            years.extend(range(int(start), int(end) + 1))
        elif token.isdigit():
            years.append(int(token))
    return sorted(list(set(years)))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Freddie Mac SFLLD Universal Step 0 Parquet Generator"
    )
    parser.add_argument(
        "--input-dir", "-i", default=".", help="Directory containing raw ZIP files"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./parquet_output",
        help="Output directory for Parquet tables",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["2006-2008", "2013-2017", "2018-2019", "2021-2022"],
    )
    parser.add_argument("--threads", "-t", type=int, default=8)
    parser.add_argument("--zstd-level", "-z", type=int, default=3)
    args = parser.parse_args()

    target_years = parse_vintage_groups(args.groups)
    logger.info(f"Target Vintages for Processing: {target_years}")

    for year in target_years:
        zip_candidate = os.path.join(args.input_dir, f"historical_data_{year}.zip")
        out_parquet = os.path.join(args.output_dir, f"vintage_{year}.parquet")

        # Recursive search across subdirectories if not at the root of input-dir
        if not os.path.exists(zip_candidate):
            found = False
            for root, _, files in os.walk(args.input_dir):
                for f in files:
                    if f == f"historical_data_{year}.zip" or (
                        str(year) in f and f.endswith(".zip")
                    ):
                        zip_candidate = os.path.join(root, f)
                        found = True
                        break
                if found:
                    break
            if not found:
                logger.warning(
                    f"SKIPPING {year}: No archive found for {year} in {args.input_dir}"
                )
                continue

        process_single_vintage(
            zip_candidate,
            out_parquet,
            threads=args.threads,
            zstd_level=args.zstd_level,
        )


if __name__ == "__main__":
    main()