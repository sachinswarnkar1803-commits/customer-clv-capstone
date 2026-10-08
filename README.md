# Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments

**Course / Programme**: T.Y. B.Sc. Data Science – Semester V  
**Project Code**: BDS-34  
**Project Creators / Team**:
- **Sachin Swarnkar** (Roll No. `TDDS028B`)
- **Shivam Yadav** (Roll No. `TDDS044B`)  
**Status**: Industry-Oriented Capstone Implementation  

> **Project Attribution**: This capstone project is made by **Sachin Swarnkar** (Roll No. `TDDS028B`) and team partner **Shivam Yadav** (Roll No. `TDDS044B`).  

---

## 1. Executive Summary & Problem Statement

Customer Lifetime Value (CLV) is a cornerstone metric for retail organizations, dictating customer acquisition costs, retention investments, and resource allocation. However, traditional heuristic approaches (such as historical average spend or simple RFM scoring) suffer from critical flaws:
1. **Survivorship & Recency Bias**: They assume past high spenders will continue indefinitely, ignoring dropout/churn risk.
2. **Deterministic False Precision**: Point-estimate CLV fails to communicate uncertainty or statistical variance.
3. **Actionability Gap**: Knowing a customer's CLV does not inform *what marketing action* or *channel* to trigger.

This capstone project implements an end-to-end, industry-standard, reproducible customer intelligence platform. Built on the official **UCI Online Retail II** dataset, the system combines:
- **Temporal Leakage-Free Data Pipeline**: Robust schema validation, audit trail, and time-based train/holdout splits.
- **Cohort Dynamics & Retention Analysis**: Acquisition month cohort matrices, repeat purchase rates, and revenue curves.
- **Probabilistic Modeling**:
  - **BG/NBD (Beta-Geometric / Negative Binomial Distribution)**: Probabilistic repeat transaction counts and $P(\text{Alive})$.
  - **Gamma-Gamma Model**: Probabilistic expected average monetary basket spend.
  - **Survival Analysis (Kaplan-Meier & Weibull Hazard)**: Customer inactivity probabilities over 30, 60, and 90-day horizons.
- **CLV Uncertainty Quantification**: 80% Monte Carlo predictive intervals ($[CLV_{lower}, CLV_{upper}]$) across multiple horizons (30, 90, 180, 365 days).
- **Action-Oriented Segmentation & Next-Best-Action (NBA) Engine**: Translating probabilistic distributions into high-leverage business actions (`RETENTION`, `WIN_BACK`, `UPSELL`, `CROSS_SELL`, `LOYALTY_REWARD`, `NO_ACTION`) with prioritized channels and estimated touch costs.
- **Interactive Campaign Scenario Simulator**: What-if ROI modeling under parameterized budget, discount, and uplift assumptions.
- **Interactive Multi-Page Streamlit Dashboard**: Professional interactive decision workspace for executive oversight, customer drill-down, and model monitoring.

---

## 2. System Architecture

```mermaid
graph TD
    A[UCI Online Retail II .zip / .xlsx] --> B[Data Ingestion: src/data/ingest.py]
    B --> C[Data Validation & Audit Trail: src/validation/validator.py]
    C --> D[Clean Analytical Dataset: data/processed/clean_transactions.parquet]
    D --> E[Temporal Splitter & Feature Engineering: src/features/]
    E --> F[Cohort & Retention Matrix: src/cohort/]
    E --> G[RFM Baseline Model: src/models/baseline.py]
    E --> H[BG/NBD Repeat Purchase Model: src/models/purchase_model.py]
    E --> I[Gamma-Gamma Monetary Model: src/models/monetary_model.py]
    E --> J[Survival / Inactivity Model: src/models/inactivity_model.py]
    H & I & J --> K[Probabilistic CLV & Uncertainty: src/clv/clv_calculator.py]
    K --> L[Action Segmentation: src/segmentation/segmenter.py]
    L --> M[Next-Best-Action Decision Engine: src/nba/nba_engine.py]
    M --> N[Scenario Simulator: src/simulation/campaign_simulator.py]
    N --> O[Evaluation & Monitoring: src/evaluation/ & src/monitoring/]
    O --> P[Interactive Streamlit Dashboard: dashboard/app.py]
```

### Architecture Diagram Generation with Archify

```bash
npx skills add tt-ali/archify -g
```

▸ More installation options and update-check details

---

### 2. Start from a description — no repository required

```text
Use Archify to draw: Browser -> API -> Redis cache -> PostgreSQL fallback.
```

For source evidence, open a repository and ask:

> Analyze this repository, then use archify to create a high-level runtime architecture diagram. Show 8–12 core components, one primary path, external dependencies, and trust boundaries. Put supporting detail in cards instead of adding more edges.

---

### 3. Refine in chat

Continue with focused requests such as `add Redis`, `move auth to the left`, or `highlight the rollback path`. Archify keeps the typed source available for targeted iteration.

---

### Choose the right diagram
▸ Five diagram types, architecture comparisons, and examples

---

## 3. Dataset Information & Provenance

- **Official Source**: [UCI Machine Learning Repository - Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii)
- **Direct Source URL**: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
- **Scope**: Transactions spanning December 1, 2009 to December 9, 2011 for a UK-based registered giftware retailer.
- **Size**: ~1,067,371 rows across two annual sheets (`Year 2009-2010` and `Year 2010-2011`).
- **Core Columns**: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.

---

## 4. Quick Start & Reproducibility Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Git

### Installation
1. Clone or navigate to the repository:
   ```bash
   cd customer-clv-capstone
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution Pipeline

1. **Ingest and Validate Raw Data**:
   ```bash
   python -m src.data.ingest
   python -m src.validation.validator
   ```
   *Note: If internet access is constrained, place the official `online_retail_II.xlsx` into `data/raw/` and run the command.*

2. **Execute Full End-to-End Pipeline**:
   ```bash
   python -m src.pipeline
   ```
   This generates the cleaned transaction dataset, customer behavioral features, cohort retention matrices, probabilistic models, CLV estimates with predictive intervals, action segments, and NBA recommendations.

3. **Run Automated Test Suite**:
   ```bash
   pytest tests/
   ```

4. **Launch Interactive Streamlit Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

---

## 5. Repository Structure

```
customer-clv-capstone/
├── configs/
│   └── default_config.yaml          # Centralized configuration (thresholds, paths, parameters)
├── data/
│   ├── raw/                         # Raw data (gitignored)
│   ├── processed/                   # Clean analytical & feature datasets
│   ├── synthetic/                   # Isolated synthetic campaign datasets
│   └── validation/                  # Schema validation audit records
├── docs/
│   ├── PRD.md                       # Product Requirements Document
│   ├── ARCHITECTURE.md              # System Architecture Specification
│   ├── RULES.md                     # Engineering Rules & Guidelines
│   ├── DESIGN.md                    # Technical & Algorithmic Design
│   ├── TASKS.md                     # Work Breakdown Structure & Milestones
│   ├── MEMORY.md                    # Project Memory & Architectural Context
│   ├── architecture/                # Architecture diagrams and specifications
│   ├── data_dictionary.md           # Field-by-field schema documentation
│   ├── contribution_log.md          # Student activity and peer review log
│   ├── threat_model.md              # Security, privacy, and leakage threat model
│   └── model_card.md                # ML model governance & ethical limitations
├── src/
│   ├── config/                      # Pydantic configuration loaders
│   ├── data/                        # Ingestion and raw file extraction
│   ├── validation/                  # Data quality, anomaly detection, and schema validation
│   ├── features/                    # Feature store, temporal splits, and aggregations
│   ├── cohort/                      # Monthly cohort retention and revenue dynamics
│   ├── models/                      # BG/NBD, Gamma-Gamma, Survival, and Baseline models
│   ├── clv/                         # Probabilistic CLV & predictive uncertainty bounds
│   ├── segmentation/                # Value-Risk matrix and action-oriented clustering
│   ├── nba/                         # Next-Best-Action rules and priority scoring
│   ├── simulation/                  # Campaign scenario simulator and ROI modeling
│   ├── evaluation/                  # Temporal holdout benchmarking, calibration, stability
│   ├── monitoring/                  # Data drift, PSI, and feature distribution monitoring
│   └── pipeline.py                  # End-to-end pipeline orchestrator
├── dashboard/
│   ├── app.py                       # Main Streamlit dashboard entrypoint
│   └── pages/                       # Multi-page modular UI views
├── tests/
│   ├── unit/                        # Unit tests for individual components
│   └── integration/                 # End-to-end integration tests
├── Dockerfile                       # Container deployment definition
├── docker-compose.yml               # Container orchestration
├── requirements.txt                 # Exact Python dependency pinning
└── README.md                        # Primary project documentation
```

---

## 6. Key Deliverables & Methodological Highlights

- **Temporal Strictness**: Zero future leakage. Cutoff dates cleanly split observation windows from holdout ground truth.
- **Statistical Rigor**: Verification of Gamma-Gamma independence assumptions prior to model fitting.
- **Uncertainty Quantification**: 80% Monte Carlo predictive intervals provide marketers with empirical risk bounds rather than misleading point precision.
- **Prescriptive Analytics**: Actionable transition from descriptive analytics ($CLV = £850$) to prescriptive intervention (*"High Value At Risk; Trigger Win-Back SMS with £15 discount voucher; Expected ROI = 280%"*).
- **Synthetic Campaign Isolation**: Strictly labeled simulation layer ensuring academic integrity.


## Review and reproducibility notes

- The inactivity model defines an event as reaching the configured 90-day inactivity threshold and treats customers who have not reached that threshold by the cutoff as right-censored.
- CLV uncertainty is reported as an empirical Monte Carlo predictive interval. It is not described as a bootstrap confidence interval for fitted parameters.
- Segment stability is evaluated with a customer-level transition matrix and agreement rate.
- `requirements.txt` uses exact package versions for reproducible environments.
- CI runs tests, Python compilation, `pip-audit`, and `bandit`.
- Docker starts the dashboard from generated artifacts when available; if raw data is present but artifacts are missing, it can run the pipeline automatically.
- The Streamlit application is an analytical capstone prototype and does not claim production authentication/RBAC.
- Campaign response and uplift inputs in the simulator are synthetic planning assumptions, not observed campaign results.

## Interactive Web Application

The project includes a professional Streamlit web application with a dedicated **Customer Action Studio** for one-to-one model interaction.

### Start the application

From the project root:

```bash
python scripts/run_dashboard.py
```

Or:

```bash
streamlit run dashboard/app.py
```

### One-customer model test

Open **Customer Action Studio** in the sidebar. Select a customer, edit the observed behavioural inputs if you want to test sensitivity, and click **Score Customer**.

The interface returns:

- P(Alive) from BG/NBD
- Expected future purchases
- Gamma-Gamma expected monetary value
- CLV prediction and uncertainty interval
- Weibull inactivity probability
- Inactivity risk tier
- Action segment
- Next-best-action
- Recommended channel
- Expected incremental value

The form does not modify the stored dataset.

### Command-line model test

After running the pipeline:

```bash
python scripts/score_customer.py --customer-id YOUR_CUSTOMER_ID
```

### Campaign Scenario Simulator

The Campaign Scenario Simulator is part of the same Streamlit application. It uses the repository root for imports, so it can be launched from the project root without a `src` import error.

The simulator is explicitly a **what-if planning tool**. Response rate, incremental AOV, discount, and costs are user-provided assumptions; they are not claimed to be observed campaign results.

---

## Authors & Contributors

This capstone project is developed by:
- **Sachin Swarnkar** — Roll No. `TDDS028B`
- **Shivam Yadav** — Roll No. `TDDS044B`

T.Y. B.Sc. Data Science – Semester V | Project Code: BDS-34
