# System & Algorithmic Design Document

**Project Name**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Project Code**: BDS-34  
**Course / Programme**: T.Y. B.Sc. Data Science – Semester V  
**Project Creators / Team**:
- **Sachin Swarnkar** (Roll No. `TDDS028B`)
- **Shivam Yadav** (Roll No. `TDDS044B`)  
**Status**: Technical Design Specification  
**Version**: 1.0.0  

---

## 1. Algorithmic Formulations & Statistical Design

### 1.1 BG/NBD Repeat Purchase Model
The BG/NBD (Beta-Geometric / Negative Binomial Distribution) model represents non-contractual continuous purchasing behavior under two core assumptions:
1. **Transaction Process**: While active, customer transaction counts follow a Poisson process with transaction rate $\lambda$. Across customers, $\lambda$ follows a Gamma distribution with shape $r$ and scale $\alpha$:
   $$\lambda \sim \text{Gamma}(r, \alpha)$$
2. **Dropout Process**: After any transaction, a customer may defect with probability $p$. The dropout opportunity follows a Geometric distribution. Across customers, $p$ follows a Beta distribution with parameters $a$ and $b$:
   $$p \sim \text{Beta}(a, b)$$

#### Probability of Being Active $P(\text{Alive})$:
Given observed frequency $x$, recency $t_x$, and observation length $T$:
$$P(\text{Alive} \mid x, t_x, T, r, \alpha, a, b) = \left[ 1 + \frac{a}{b + x - 1} \left( \frac{\alpha + T}{\alpha + t_x} \right)^{r + x} \right]^{-1}$$

#### Expected Transactions $E[Y(t)]$:
Over a future horizon $t$:
$$E[Y(t) \mid x, t_x, T] = \frac{a + b + x - 1}{a - 1} \left[ 1 - \left( \frac{\alpha + T}{\alpha + T + t} \right)^{r + x} \cdot \frac{{}_2F_1\left(r + x, b + x; a + b + x - 1; \frac{t}{\alpha + T + t}\right)}{1} \right] \cdot P(\text{Alive})$$

---

### 1.2 Gamma-Gamma Monetary Model
The Gamma-Gamma model characterizes customer basket monetary value:
1. Average transaction value $\bar{m}$ for each customer is normally distributed around expected value $\mu$ with Gamma shape $p$.
2. Across customers, $\mu$ follows a Gamma distribution with shape $q$ and scale $v$.
3. Frequency $x$ and transaction value $\bar{m}$ are assumed independent.

#### Expected Monetary Value $E[M]$:
$$E[M \mid p, q, v, x, \bar{m}] = \frac{q - 1}{p \cdot x + q - 1} \cdot \frac{v \cdot p}{q - 1} + \frac{p \cdot x}{p \cdot x + q - 1} \cdot \bar{m}$$

---

### 1.3 Time-to-Inactivity Survival Analysis
- **Non-Parametric**: Kaplan-Meier product-limit estimator:
  $$\hat{S}(t) = \prod_{t_i \le t} \left( 1 - \frac{d_i}{n_i} \right)$$
- **Parametric**: Weibull survival distribution with scale $\lambda$ and shape $k$:
  $$S(t) = \exp\left( -(\lambda t)^k \right)$$
  $$\text{Inactivity Risk at horizon } h = 1 - S(h \mid T_{split} - t_x)$$

---

### 1.4 Discounted CLV & Monte Carlo Predictive Intervals
For a customer with $E[Y(h)]$ expected purchases over horizon $h$ and discount rate $\delta$:
$$\text{CLV}(h) = \sum_{t=1}^h \frac{E[Y(t)] \cdot E[M]}{(1 + \delta / 365)^t}$$

To compute 80% predictive intervals $[CLV_{lower}, CLV_{upper}]$, we draw $N=1,000$ Monte Carlo samples from the joint posterior distributions of transaction rate $\lambda$ and monetary value $\mu$.

---

## 2. Decision Logic: Next-Best-Action (NBA) Engine

| Segment | Primary Condition | Recommended Action | Default Channel | Unit Touch Cost | Expected Lift |
|---|---|---|---|---|---|
| `CHAMPION_HIGH_VALUE` | High CLV, $P(\text{Alive}) \ge 0.85$ | `LOYALTY_REWARD` | `EMAIL` / Exclusive Call | £1.50 | +12% AOV |
| `LOYAL_CORE` | Moderate CLV, $P(\text{Alive}) \ge 0.70$ | `UPSELL` | `EMAIL` | £0.50 | +8% Freq |
| `POTENTIAL_LOYALIST` | Low Freq, Recent First Buy | `CROSS_SELL` | `EMAIL` / SMS | £0.75 | +15% Freq |
| `AT_RISK_HIGH_VALUE` | High CLV, $0.35 \le P(\text{Alive}) < 0.70$ | `RETENTION` | `ACCOUNT_CALL` / Direct Mail | £4.00 | +25% Retention |
| `HIBERNATING` | Low CLV, $P(\text{Alive}) < 0.35$ | `WIN_BACK` | `EMAIL` | £0.20 | +5% Return |
| `LOST_CUSTOMER` | Inactive $> 180$ days, $P(\text{Alive}) < 0.15$ | `NO_ACTION` | `NONE` | £0.00 | 0% |

---

## 3. UI/UX & Streamlit Application Design

### 3.1 Design Principles
- **No-Emoji Executive Aesthetic**: Clean typography, crisp tabular metrics, and enterprise styling.
- **Immediate Interactivity**: Single-customer scoring updates in real-time ($< 200\text{ms}$).
- **Action-First Hierarchy**: Insights always pair with actionable recommendations.

### 3.2 Page Hierarchy
1. **Executive Dashboard**: Portfolio summary metrics, aggregate CLV distribution, active customer counts.
2. **Cohort Dynamics Studio**: Monthly retention heatmap, cumulative revenue curve.
3. **Customer Action Studio**: Individual customer lookup, scenario parameter override, real-time prediction and Next-Best-Action decision card.
4. **Campaign Scenario Planner**: Interactive sliders for targeted segment, response rate, discount, touch costs, with ROI and breakeven visualization.
5. **Model Health & Monitoring**: Calibration plots, PSI drift alerts, and holdout error metrics.
