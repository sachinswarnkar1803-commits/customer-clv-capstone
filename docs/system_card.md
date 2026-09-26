# System Card: BDS-34 Customer Intelligence Platform

**System Name**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Course**: T.Y. B.Sc. Data Science – Semester V  
**Status**: Capstone Production-Grade Implementation  

---

## 1. System Architecture & Component Mapping

The system operates as a modular, end-to-end data science application:

| System Layer | Sub-Modules | Input Artifacts | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `src/data/ingest.py` | Official UCI zip archive | `data/raw/online_retail_II.xlsx` |
| **Validation** | `src/validation/validator.py` | Raw Excel sheets | `clean_transactions.parquet`, `audit_trail.json` |
| **Feature Store** | `src/features/` | Clean transactions | `customer_features.parquet`, RFM scores |
| **Cohort Dynamics** | `src/cohort/` | Transaction timestamps | `cohort_retention_matrix.csv`, `cohort_summary.csv` |
| **Statistical Models**| `src/models/` | RFM & tenure matrices | Serialized BG/NBD, Gamma-Gamma, Survival models |
| **CLV Engine** | `src/clv/` | Model instances | `clv_expected_90d`, 80% bootstrap intervals |
| **Prescriptive Engine**| `src/segmentation/`, `src/nba/` | Probabilistic CLV & risk | `customer_clv_segments.parquet` |
| **Simulation** | `src/simulation/` | Campaign parameters | Projected revenues, costs, ROI distributions |
| **Evaluation** | `src/evaluation/` | Holdout transactions | `model_evaluation_report.json` |
| **Monitoring** | `src/monitoring/` | Temporal partitions | Population Stability Index (PSI), drift metrics |
| **Interface** | `dashboard/app.py` | Processed Parquet tables | Interactive Streamlit command center |

---

## 2. Hardware and Environment Requirements
- **Runtime**: Python 3.10+ (tested on Python 3.11.0)
- **RAM**: Minimum 4GB (8GB recommended for full 1M row Excel parsing)
- **Disk Space**: ~250MB (including raw data and processed Parquet caches)
- **Container**: Multi-stage Linux Debian slim container (`Dockerfile`)

---

## 3. Maintenance, Monitoring & Drift Thresholds
- **Population Stability Index (PSI)**:
  - $PSI < 0.10$: Healthy (No action required)
  - $0.10 \le PSI < 0.25$: Warning (Moderate distribution shift; schedule model inspection)
  - $PSI \ge 0.25$: Alert (Significant concept drift; trigger automated retraining pipeline)
- **Data Quality Alerts**: Logged non-silently in `data/validation/audit_trail.json`. Any sudden jump in unauthenticated checkouts or negative price line items triggers an alert.
