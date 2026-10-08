# Project Memory & Architectural Context (MEMORY.md)

**Project Name**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Project Code**: BDS-34  
**Course / Programme**: T.Y. B.Sc. Data Science – Semester V  
**Project Creators / Team**:
- **Sachin Swarnkar** (Roll No. `TDDS028B`)
- **Shivam Yadav** (Roll No. `TDDS044B`)  
**Status**: Persistent System Knowledge Base  
**Version**: 1.0.0  

---

## 1. Project Context & Domain Facts

- **Domain**: Non-contractual continuous retail (giftware e-commerce).
- **Core Dataset**: UCI Machine Learning Repository *Online Retail II* (December 1, 2009 – December 9, 2011).
- **Dataset Size**: ~1,067,371 rows across two Excel sheets (`Year 2009-2010` and `Year 2010-2011`).
- **Data Quirks & Handling**:
  - Invoices starting with `C` represent credit/cancellation orders and must be matched or purged from net positive spend computations.
  - Rows with missing `Customer ID` (~20% of raw data) represent guest checkouts and cannot be attributed to repeat behavioral models; they are tracked in data quality reports but excluded from customer-level RFM modeling.
  - Extreme outliers (e.g. quantity $> 10,000$ or unit price $< 0$) are filtered by schema validation contracts.

---

## 2. Architectural Decision Records (ADRs)

### ADR-01: Temporal Splitting vs Random Train/Test Split
- **Decision**: Partition data chronologically at $T_{split} = \text{2010-12-09}$ instead of a random $80/20$ customer split.
- **Rationale**: Retail transaction behavior exhibits strong seasonality, holiday spikes, and recency decay. Random partitioning leaks future customer actions into past feature calculations, giving falsely optimistic accuracy.

### ADR-02: Monte Carlo Predictive Intervals vs Bayesian Credible Intervals
- **Decision**: Generate predictive uncertainty intervals via Monte Carlo parameter simulation from MLE fits.
- **Rationale**: Full MCMC Bayesian inference for 5,000+ customers takes orders of magnitude longer to compute. Monte Carlo draws from estimated parameter variance provide rapid, calibrated uncertainty intervals suitable for interactive dashboards.

### ADR-03: Right-Censoring in Inactivity Survival Modeling
- **Decision**: Use a right-censored event framework where customers still active at $T_{split}$ are censored ($E=0$) with duration $T_{split} - \text{FirstPurchase}$.
- **Rationale**: Marking non-churned customers as inactive or dropping them biases survival curves toward hyper-accelerated churn.

### ADR-04: Rule-Governed Next-Best-Action Engine vs Black-Box Optimization
- **Decision**: Implement a transparent, rule-governed Next-Best-Action engine with parameterized channel costs and expected uplifts.
- **Rationale**: Enterprise marketing and retention managers require explainable decisions for why a customer received a specific offer and channel.

---

## 3. Important Gotchas & Operational Notes

1. **Dashboard Execution Path**:
   - The Streamlit application in `dashboard/app.py` references root-level imports (`src.models`, etc.).
   - Always run via `python scripts/run_dashboard.py` or `streamlit run dashboard/app.py` from the **project root directory** (`customer-clv-capstone/`).

2. **Model Retraining & Parquet Files**:
   - Intermediate Parquet files (`clean_transactions.parquet`) are used to accelerate pipeline execution.
   - If raw data is updated, delete the files in `data/processed/` and run `python -m src.pipeline` to rebuild the pipeline end-to-end.

3. **Lifecycle Model Assumptions**:
   - The Gamma-Gamma model requires customers to have at least one repeat purchase ($x \ge 1$). One-time purchasers are assigned the population average monetary value until second purchase.
   - Pearson correlation between frequency and monetary value should be small ($|r| < 0.20$); Online Retail II exhibits $r \approx -0.05$, satisfying the independence assumption.

---

## 4. System Directory Map

```
customer-clv-capstone/
├── configs/               # Hyperparameters, split dates, and model configs
├── dashboard/             # Streamlit interactive application
├── data/
│   ├── raw/               # Downloaded UCI raw files
│   ├── processed/         # Cleaned Parquet transaction files
│   └── validation/        # Schema contracts and validation rules
├── docs/                  # Architecture, PRD, Rules, Design, Tasks, Memory
├── models/                # Trained model artifacts (BG/NBD, Gamma-Gamma, Weibull)
├── notebooks/             # Exploratory analysis and modeling notebooks
├── reports/               # Quality, stability, and calibration reports
├── scripts/               # CLI tools, dashboard runner, performance checks
├── src/                   # Core pipeline modules
└── tests/                 # Unit and regression tests
```
