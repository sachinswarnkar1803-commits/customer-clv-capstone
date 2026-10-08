# User Guide: BDS-34 Customer Intelligence Platform

Welcome to the **Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments** platform. This guide explains how to install, run, and navigate the system.

---

## 1. Quick Setup & Pipeline Execution

### Step 1: Clone and Install
```bash
git clone <repo-url>
cd customer-clv-capstone
pip install -r requirements.txt
```

### Step 2: Ingest Data & Execute Pipeline
To download the dataset and execute the entire pipeline:
```bash
python -m src.pipeline
```
This runs the full 13-stage workflow:
1. Validates and cleans 1,067,371 raw records.
2. Partitions data temporally (Cutoff: `2010-12-09`).
3. Fits BG/NBD, Gamma-Gamma, and a time-to-inactivity Weibull survival model.
4. Generates 80% Monte Carlo CLV predictive intervals.
5. Classifies customers into action segments.
6. Computes Next-Best-Actions (NBA), channels, and offers.
7. Evaluates predictions against real holdout outcomes.
8. Computes Population Stability Index (PSI) drift indicators.

### Step 3: Run Automated Test Suite
```bash
pytest
```
Run the automated test suite; CI is the authoritative environment for the full dependency-backed test result.

### Step 4: Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 2. Navigating the Dashboard

| Tab / Page | Key Functions & Business Use Cases |
| :--- | :--- |
| **📊 Executive Overview** | High-level KPI tiles (Active Customers, Total Spend, Avg CLV, Champions, Risk Count), segment breakdown, and value-risk strategic scatter plot. |
| **👥 Cohort & Retention** | Interactive monthly retention matrix heatmap, cohort survival curves, and cumulative revenue tracking across acquisition cohorts. |
| **🔍 Customer 360 Explorer**| Drill down into any customer by ID: see country, tenure, P(Alive), 90d Inactivity Hazard, Expected CLV with 80% predictive intervals, and full order basket history. |
| **🎯 Action Segments** | Audit the 8 operational segments (Champions, High Value At Risk, Growing, Loyal Mid Value, New Customer, Reactivation Candidate, Low Value Active, Dormant). |
| **⚡ Next-Best-Action** | Operational action queue ranked by priority and expected incremental return (£). Guides marketers on channel, offer, and contact cost. |
| **🎲 Campaign Simulator** | Interactive what-if ROI calculator. Test budget, response rate, and discount assumptions before running campaigns. |
| **🛡️ Model Monitoring** | Auditing holdout error benchmarking (MAE, RMSE, Rank Correlation), 80% interval empirical coverage, calibration Brier score, and PSI drift alerts. |
