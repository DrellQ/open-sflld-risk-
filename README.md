# open-sfld-risk

### High-Level Data & Compute Flow

```
                          Freddie Mac SFLLD (18M+ Loans across 4 Regimes)
                                                │
                                                ▼
                               Polars Ingestion & Preprocessing
                              (Sentinel Purging, Clean Horizon t=12)
                                                │
                                                ▼
                            Domain-Constrained Monotonic XGBoost
                              (Negative FICO, Positive DTI/LTV)
                                                │
                                                ▼
                           Out-of-Time Platt Scaling Calibration
                                (Brier: 0.138 ➔ 0.020 Parity)
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
           Macroeconomic Stress Testing                      Vectorized Monte Carlo VaR
           • GFC Stress Anchor (2006–2008)                   • 25,000 Stochastic Iterations
           • Empirical Downturn LGD (33.31%)                 • $1.51T Active UPB Exposure
           • Rate Shock Cohort (2021–2022)                   • EL, VaR(95/99/99.9), ES(99)
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                                   Synthetic CRT / STACR Engine
                                 (B-2, B-1, M-2, M-1, A-H Waterfall)
```

---

## Some Design Details

### A. Non-Look-Ahead Temporal Horizon Framing
* **Design**: All features are anchored at loan observation snapshot month 12. Default is defined as reaching 90+ days delinquent, foreclosure, or liquidation strictly within the forward 12 month horizon 13-24.
* **Explanation**: Eliminates look ahead leakage. Demanding loans to be active and non-delinquent at 12 replicates the real world operational monitoring cycle of a GSE or bank balance sheet.

### B. Domain-Constrained Monotonic Gradient Boosted Trees
* **Design**: Enforced strict monotonic splits in XGBoost:
  * `credit_score`: Monotonic constraint **-1** 
  * `original_ltv`, `original_cltv`, `original_dti`, `original_interest_rate`, `snapshot_dq`, `upb_paydown_ratio`: Monotonic constraint **+1** 
* **Explanation**: Standard unconstrained gradient trees often discover spurious, non linear splits in sparse feature regions (e.g., scoring high FICO borrowers as higher risk). Monotonicity guarantees economically sound behavior.

### C. Out-of-Time Platt Probability Calibration
* **Design**: Applied univariate logistic scaling (Platt scaling) fitted on the 2018–2019 Out-of-Time cohort
* **Explanation**: Training with `scale_pos_weight = 230.65` offset severe class imbalance (0.43% event rate) but expanded raw outputs into uncalibrated decision margins. Platt scaling recovers empirical probability parity while perfectly preserving rank discrimination (ROC-AUC).

### D. Empirical Downturn LGD Parameterization
* **Design**: Loss Given Default parameters were mined directly from 3,848 realized liquidations in the 2006–2008 Great Financial Crsis.
* **Explanation**: Replaces arbitrary regulatory assumptions (such as static 40% haircuts) with empirically verified loss severity distributions observed during severe house price depreciation and secondary-market illiquidity.

### E. Vectorized Monte Carlo Simulation Engine
* **Design**: Executed a 25,000-path stochastic simulation drawing Bernoulli default outcomes and truncated normal LGD shocks across stratified loan batches
* **Explanation**: Allows modeling of non-linear loss clustering, fat-tailed credit distributions


---
# Source Files Overview

### 1. `src/feature_engineering.py`
* **Purpose**: Primary data transformation, schema alignment, and missing value cleaning.
* **Key Tasks**:
  * Purges Freddie Mac missing-values (`9999` for credit score, `999.0` for LTV/CLTV/DTI) and converts them to native nulls so XGBoost routes them to default decision splits.
  * Enforces the observation snapshot at 12m, deriving `snapshot_dq`, `upb_paydown_ratio`, and forward default outcome `y_target` (13-24).
  * Enforces categorical domain consistency across historical origination channels.

### 2. `src/train_xgboost.py`
* **Purpose**: Model training, monotonic constraint enforcement, and baseline validation.
* **Key Tasks**:
  * Trains the primary XGBoost classifier on 6.98M loans from the 2013–2017 QM baseline.
  * Applies `monotone_constraints` (`-1` for credit score, `+1` for leverage and delinquency metrics).
  * Computes out-of-time decile rank-ordering tables verifying monotonic decay in default rates from Decile 1 down to Decile 10.
  * Persists the serialized model to `models/xgboost_sflld_stage1.json` and logs output to `data validation reports/`.

### 3. `src/calibrate_and_plot.py`
* **Purpose**: Out-of-time probability calibration and visual diagnostic curve generation.
* **Key Tasks**:
  * Fits a Platt calibrator on the 2018–2019 Out of Time cohort to correct positive class weighting distortion.
  * Serializes the calibrator to `models/platt_calibrator_stage1.joblib`.
  * Generates and exports plots to `reports/figures/model_calibration_and_roc.png` (Reliability Diagram, ROC Curve, and Score Distribution Shift).

### 4. `src/stress_test_var.py`
* **Purpose**: Macro stress evaluation, downturn LGD mining, and portfolio VaR simulation.
* **Key Tasks**:
  * Evaluates model discrimination against the 2006–2008 GFC Anchor and the 2021–2022 Rate Shock cohort.
  * Mines 3,848 realized liquidations to compute empirical downturn LGD parameters
  * Executes the Monte Carlo VaR engine using Platt adjusted default probabilities,
  calculating EL, VaR(95/99/99.9), Expected Shortfall, and Economic Capital.

### 5. `src/generate_portfolio_loss_distribution.py`
* **Purpose**: Visualization of portfolio loss density and capital thresholds.
* **Key Tasks**:
  * Executes a 25,000-path Monte Carlo credit loss simulation.
  * Generates `reports/figures/portfolio_var_loss_distribution.png`, illustrating:
    * The smooth loss density curve centered at Expected Loss ($8.68B).
    * Vertical reference lines for VaR (95%, 99%, 99.9%) and Expected Shortfall (99%).
    * Red tail-risk shading for losses exceeding 99.0% VaR, visually highlighting the $900M Unexpected Loss / Economic Capital requirement.

### 6. `src/crt_tranche_waterfall.py`
* **Purpose**: Structured finance loss allocation across synthetic STACR tranches.
* **Key Tasks**:
  * Configures attachment and detachment points for Class B-2, B-1, M-2, M-1, and Senior A-H tranches.
  * Routes simulated portfolio loss paths sequentially from subordinate to senior tranches.
  * Computes tranche Expected Loss, percentage loss severity, and 99.0% VaR.

### 7. `src/universal_sflld_etl.py`
* **Purpose**: takes the raw origination and performance data directly from the compreszed zip files and transfers them into Parquet format using a zero-disk in-memory streaming pipeline for better performance and reduced disk size. 

### 8. `src/parquet_validation.py`
* **Purpose**: Validates schema compliance, sentinel codes, distribution bounds,
and credit target mechanics prior to XGBoost and Vasicek correlation training.


---