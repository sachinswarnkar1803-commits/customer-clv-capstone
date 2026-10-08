# Product Requirements Document (PRD)

**Project Name**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Project Code**: BDS-34  
**Course / Programme**: T.Y. B.Sc. Data Science – Semester V  
**Project Creators / Team**:
- **Sachin Swarnkar** (Roll No. `TDDS028B`)
- **Shivam Yadav** (Roll No. `TDDS044B`)  
**Status**: Production-Ready Capstone Deliverable  
**Version**: 1.0.0  

---

## 1. Executive Summary & Vision

In enterprise retail and e-commerce, generic customer relationship management (CRM) models rely predominantly on static, historical metrics such as total spend or heuristic Recency-Frequency-Monetary (RFM) scoring. These methods are inherently vulnerable to survivorship bias, fail to quantify variance or uncertainty in future value, and create an "actionability gap" where commercial teams do not know which marketing action or channel to trigger.

The **Customer CLV & Next-Best-Action Intelligence Platform** bridges this divide by providing an end-to-end, statistically sound, and transparent decision-support system. It transforms raw transaction histories from the UCI Online Retail II dataset into probabilistic customer valuations, right-censored churn/inactivity risk estimates, action-oriented segments, and individualized Next-Best-Action (NBA) interventions with simulated ROI.

---

## 2. Target Stakeholders & Personas

| Persona | Role | Primary Needs | Supported Decision |
|---|---|---|---|
| **CRM & Retention Lead** | Retention Operations | Identify high-value customers showing signs of defection before churn occurs. | Targeted retention discounts and proactive outreach. |
| **Marketing & Growth Analyst** | Strategic Planning | Understand customer lifetime distributions, cohorts, and expected future revenue. | Budget allocation across acquisition channels and LTV-based planning. |
| **Campaign Manager** | Operational Marketing | Receive concrete actions, channels, estimated costs, and expected returns. | Automated batch campaign execution and channel prioritization. |
| **Lead Data Scientist** | Statistical Governance | Ensure temporal separation, baseline benchmarking, and rigorous calibration. | Model retraining cycles, feature drift auditing, and calibration checks. |
| **Academic / Industry Reviewer** | Quality & Verification | Validate reproducibility, test coverage, methodological rigor, and authorship. | Project evaluation, grading, and architectural review. |

---

## 3. Product Scope & Dataset Boundary

- **Dataset**: UCI Machine Learning Repository — *Online Retail II* (December 2009 to December 2011).
- **Domain**: Non-contractual, continuous giftware retail transactions.
- **Population**: ~1.06M transaction records representing ~5,800 unique customers across 43 countries.
- **Scope Inclusions**:
  - Automated data ingestion, cleaning, validation, and contract auditing.
  - Leakage-safe temporal splitting (Train: Dec 2009 – Dec 2010; Holdout: Dec 2010 – Dec 2011).
  - Cohort retention matrices and revenue dynamics.
  - Probabilistic repeat purchase modeling (BG/NBD) and monetary modeling (Gamma-Gamma).
  - Time-to-inactivity survival modeling (Kaplan-Meier and Weibull hazard models).
  - Multi-horizon CLV calculation (30, 90, 180, 365 days) with Monte Carlo predictive intervals.
  - Action-oriented segmentation and Next-Best-Action (NBA) decision engine.
  - What-If Campaign Scenario Simulator with cost/uplift sensitivity analysis.
  - Interactive multi-page Streamlit decision workspace.
- **Scope Exclusions**:
  - Direct live third-party email/SMS API dispatch (mocked via structured queue export).
  - Multi-tenant enterprise SSO (analytical customer IDs used; authentication out-of-scope).

---

## 4. Functional Requirements (FR)

### FR-01: Ingestion & Schema Contract Enforcement
- The system must download or ingest raw `.xlsx` / `.zip` / `.csv` transaction datasets.
- Schema contract validation must ensure non-null customer IDs, valid ISO dates, positive unit prices, and correct handling of credit cancellations (`C` prefix).
- Generates an immutable data audit trail log.

### FR-02: Leakage-Safe Feature Engineering & Temporal Splitting
- All predictive modeling features must be derived strictly prior to a configurable temporal cutoff date $T_{split}$.
- Calculates customer RFM parameters ($x, t_x, T, \bar{M}$) without future information leakage into training sets.

### FR-03: Cohort Dynamics & Retention Curves
- Generates monthly customer acquisition cohorts.
- Computes period-over-period retention heatmaps, cumulative cohort revenue, and repeat purchase progression.

### FR-04: Probabilistic Transaction Modeling (BG/NBD)
- Fits Beta-Geometric/Negative Binomial Distribution parameters ($r, \alpha, a, b$) via Maximum Likelihood Estimation (MLE).
- Computes individual customer probability of being active $P(\text{Alive})$ and expected transactions $E[Y(t)]$ over defined future time horizons.

### FR-05: Probabilistic Monetary Modeling (Gamma-Gamma)
- Validates the independence assumption between transaction frequency and average monetary value.
- Estimates expected conditional monetary spend $E[M]$ per transaction using Gamma-Gamma distribution parameters ($p, q, v$).

### FR-06: Time-to-Inactivity Survival Analysis
- Implements non-parametric Kaplan-Meier survival curves and parametric Weibull hazard models.
- Formulates proper right-censoring for active customers to predict survival and inactivity probabilities over 30, 60, and 90 days.

### FR-07: Probabilistic CLV Calculation & Monte Carlo Uncertainty
- Calculates discounted Customer Lifetime Value across 30, 90, 180, and 365 days with annualized discount rate $\delta$.
- Computes 80% Monte Carlo predictive intervals $[CLV_{lower}, CLV_{upper}]$ reflecting parameter and sampling uncertainty.

### FR-08: Action-Oriented Segmentation & Next-Best-Action (NBA) Engine
- Classifies customers into 6 distinct behavioral tiers:
  - `CHAMPION_HIGH_VALUE`, `LOYAL_CORE`, `POTENTIAL_LOYALIST`, `AT_RISK_HIGH_VALUE`, `HIBERNATING`, `LOST_CUSTOMER`.
- Determines deterministic NBA strategies (`RETENTION`, `WIN_BACK`, `UPSELL`, `CROSS_SELL`, `LOYALTY_REWARD`, `NO_ACTION`).
- Maps each action to prioritized channels (`EMAIL`, `SMS`, `DIRECT_MAIL`, `ACCOUNT_CALL`, `NONE`) with touch costs and expected incremental uplift.

### FR-09: Campaign Scenario Simulator & What-If Planner
- Provides an interactive business planning tool to simulate campaign ROI.
- Enables parameter tuning: target segment, response rate, discount percentage, touch cost, and incremental basket uplift.
- Outputs net ROI, campaign budget required, incremental revenue, and breakeven response rates.

### FR-10: Baseline Benchmarking & Model Evaluation
- Compares probabilistic models against traditional heuristic RFM baselines on the holdout test dataset.
- Evaluates holdout Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Spearmans rank correlation, and coverage of predictive intervals.

### FR-11: Interactive Streamlit Decision Workspace
- Provides executive KPI overviews, single-customer drill-down scoring ("Customer Action Studio"), scenario simulation, and drift monitoring dashboards.

---

## 5. Non-Functional Requirements (NFR)

- **NFR-01: Reproducibility**: Deterministic random seeds (`seed=42`) across all Monte Carlo simulations and train/test splits.
- **NFR-02: Performance**: End-to-end pipeline execution within 180 seconds on standard commodity hardware. Interactive dashboard query latency $< 200\text{ms}$.
- **NFR-03: Security & Code Quality**: Zero hardcoded secrets, passing static code security analysis (`bandit`), and dependency audit (`pip-audit`).
- **NFR-04: Portability**: Fully containerized via multi-stage `Dockerfile` and `docker-compose.yml`.
- **NFR-05: Maintainability**: Standardized typing, docstrings, modular code structure, and automated CI test suite via GitHub Actions.

---

## 6. Success Metrics & KPIs

| Metric | Target | Verification Method |
|---|---|---|
| **Holdout CLV Ranking Correlation** | Spearman $\rho > 0.65$ | Evaluated against 1-year holdout actual revenue |
| **Predictive Interval Calibration** | 80% Monte Carlo Interval Coverage $\ge 75\%$ | Empirical coverage on holdout observations |
| **Pipeline Reliability** | 100% automated test passing rate | Pytest suite in GitHub Actions CI |
| **Business Usability** | Full single-customer scoring $< 500\text{ms}$ | Customer Action Studio benchmark test |
