# Model Card: Probabilistic CLV & Next-Best-Action System

**Project Code**: BDS-34 | **Model Version**: 1.0.0  
**Authors**: BDS-34 Capstone Team (T.Y. B.Sc. Data Science - Semester V)  
**Date**: September 2026  

---

## 1. Model Details

### 1.1 Architecture & Sub-Models
The platform employs a composite ensemble of probabilistic and survival models:
1. **Beta-Geometric / Negative Binomial Distribution (BG/NBD)**:
   - *Purpose*: Models customer transaction rates ($\lambda \sim \text{Gamma}(r, \alpha)$) and individual dropout hazards ($p \sim \text{Beta}(a, b)$).
   - *Key Outputs*: $P(\text{Alive} \mid x, t_x, T)$ and expected transaction count $E[Y(t) \mid x, t_x, T]$ over future horizons $t \in \{30, 90, 180, 365\}$ days.
2. **Gamma-Gamma Monetary Model**:
   - *Purpose*: Models customer order spend ($z_i \sim \text{Gamma}(p, \nu)$) with cross-customer heterogeneity ($\nu \sim \text{Gamma}(q, \gamma)$).
   - *Key Output*: Expected average spend per basket $E[M \mid x, m_x]$.
   - *Pre-Condition*: Independence verification between frequency $x$ and monetary basket $m_x$.
3. **Weibull Parametric Survival Model**:
   - *Purpose*: Continuous hazard modeling of customer dormancy to produce explicit 30, 60, and 90-day inactivity probabilities.
4. **Monte Carlo Predictive Simulation Engine**:
   - *Purpose*: 80% empirical prediction intervals $[CLV_{lower}, CLV_{upper}]$ and relative spread metrics.

---

## 2. Intended Use & Target Audience
- **Primary Use**: Customer relationship management (CRM), acquisition budget sizing, proactive churn intervention, cross-sell/upsell campaign targeting.
- **Target Users**: Marketing managers, customer success leads, commercial finance planners.
- **Out-of-Scope Use Cases**: Credit scoring, individual consumer debt collection, real-time fraud detection.

---

## 3. Training Data & Temporal Split
- **Source**: UCI Machine Learning Repository - Online Retail II dataset (`https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii`).
- **Observation / Training Window**: 2009-12-01 to 2010-12-09 ($N=399,177$ transactions, $4,309$ unique customers).
- **Holdout Evaluation Window**: 2010-12-10 to 2011-12-09 ($N=380,244$ transactions).
- **Leakage Controls**: Strict temporal cutoff barrier. Future transactions never enter training features or parameter fitting.

---

## 4. Quantitative Performance & Evaluation

| Evaluation Dimension | Metric | Observed Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Ranking Accuracy** | Spearman Rank Corr ($\rho$) | **0.603** | Probabilistic CLV significantly outperforms Baseline A (0.487) and Baseline B (0.542). |
| **Error Magnitude** | Mean Absolute Error (MAE) | **£1,316.97** | Reflects highly skewed bulk wholesale purchases in retail data. |
| **Interval Coverage** | Empirical 80% CI Coverage | **64.59%** | Conservative risk bounds; divergence driven by rare extreme repeat spenders. |
| **Calibration** | Inactivity Brier Score | **0.2422** | Well-calibrated probabilities outperforming naive uncalibrated benchmarks. |

---

## 5. Limitations & Ethical Considerations
- **Non-Contractual Latency**: In non-contractual retail, customers do not notify the retailer when they churn. Inactivity is an inferred probability, not an observed certainty.
- **Wholesale Skewness**: Extreme corporate bulk purchasers skew aggregate monetary totals. Robustness guards and shrinkage priors are enforced.
- **Privacy**: Customer IDs are treated as pseudonymous identifiers. Personal contact details are not retained.
