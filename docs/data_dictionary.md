# Data Dictionary - BDS-34 Capstone Project

**Project Title**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Dataset**: Online Retail II (UCI Machine Learning Repository, ID: 502)  
**Timeframe**: 01/12/2009 – 09/12/2011  

---

## 1. Raw Dataset Schema (UCI Online Retail II)

The original source data contains transaction records from a UK-based online non-store retail business. It contains two yearly sheets: `Year 2009-2010` and `Year 2010-2011`.

| Column Name | Raw Data Type | Target Data Type | Description | Constraints & Cleaning Logic |
| :--- | :--- | :--- | :--- | :--- |
| `Invoice` | Object / String | String | 6-digit invoice number uniquely assigned to each transaction. | If it starts with 'C', it denotes a cancellation. Missing or malformed invoices flagged. |
| `StockCode` | Object / String | String | 5-digit integral number uniquely assigned to each distinct product. | May include administrative codes (e.g., `POST`, `D`, `M`, `BANK CHARGES`, `AMAZONFEE`). |
| `Description` | Object / String | String | Product / item name. | Can be null in corrupt/cancelled rows. Stripped of whitespace. |
| `Quantity` | Int64 / Float64 | Int64 | The quantities of each product per transaction. | Negative values indicate cancellations, damages, or returns. Filtered for standard CLV purchase models. |
| `InvoiceDate` | Datetime / Object | Datetime64[ns] | The day and time when each transaction was generated. | Must be within valid date window (2009-12-01 to 2011-12-09). |
| `Price` | Float64 | Float64 | Product unit price in GBP (£). | Must be > 0 for completed commercial purchases. Zero or negative prices indicate inventory adjustments or bad debts. |
| `Customer ID` | Float64 / Object | Int64 / String | 5-digit number uniquely assigned to each customer. | Null customer IDs (~20-25% of raw records) cannot be attributed to customer-level lifetimes and are tracked in audit trail. |
| `Country` | Object / String | String | Name of the country where the customer resides. | Categorical; UK represents >85% of transactions. |

---

## 2. Standardized Analytical Transaction Table (`clean_transactions`)

Generated after applying quality gates and cleaning rules.

| Column Name | Data Type | Description | Business Purpose |
| :--- | :--- | :--- | :--- |
| `customer_id` | String | Unique standardized customer identifier. | Primary join key for customer aggregation. |
| `invoice_id` | String | Unique invoice / order number. | Order grouping and basket identification. |
| `transaction_date` | Datetime64[ns] | Date & timestamp of purchase. | Temporal sequencing, cadence, and recency. |
| `stock_code` | String | Standardized product code. | Basket variety and product analytics. |
| `description` | String | Standardized product name. | Catalog audit and classification. |
| `quantity` | Int64 | Unit quantity purchased (strictly positive). | Volume analysis. |
| `unit_price` | Float64 | Price per unit in GBP (£) (strictly positive). | Pricing tiering. |
| `revenue` | Float64 | `quantity * unit_price` (£). | Primary monetary metric. |
| `country` | String | Standardized country name. | Geographic segmentation. |
| `is_return` | Boolean | True if transaction is a return/adjustment. | Return rate profiling. |
| `is_cancelled` | Boolean | True if invoice was cancelled. | Cancellation rate profiling. |
| `invoice_month` | Period[M] | Month of transaction (e.g. `2010-01`). | Monthly aggregation and cohort tracking. |

---

## 3. Customer Feature Table (`customer_features`)

Customer-level aggregated behavioral and RFM metrics respecting temporal cutoff causality.

| Field Name | Type | Formula / Definition | Strategic Meaning |
| :--- | :--- | :--- | :--- |
| `customer_id` | String | Unique customer identifier | Entity key |
| `first_purchase_date`| Datetime | $\min(\text{transaction\_date})$ | Customer acquisition moment |
| `last_purchase_date` | Datetime | $\max(\text{transaction\_date})$ | Most recent transaction moment |
| `customer_tenure_days`| Float64 | $(\text{cutoff\_date} - \text{first\_purchase\_date})$ in days | Lifetime duration under observation ($T$) |
| `recency_days` | Float64 | $(\text{last\_purchase\_date} - \text{first\_purchase\_date})$ in days | Recency ($t_x$) in lifetimes models |
| `days_since_last_purchase` | Float64 | $(\text{cutoff\_date} - \text{last\_purchase\_date})$ in days | Elapsed inactivity time |
| `frequency` | Int64 | Number of repeat purchase dates ($x$ where $x \ge 0$) | Repeat buying propensity |
| `total_invoices` | Int64 | Total distinct orders placed | Absolute order count |
| `monetary_value` | Float64 | Average basket value across repeat orders | Input $m_x$ for Gamma-Gamma model |
| `total_revenue` | Float64 | Sum of all revenue across the observation period | Historical commercial contribution |
| `average_order_value`| Float64 | `total_revenue / total_invoices` | Basket size benchmark |
| `unique_products` | Int64 | Distinct `stock_code` count | Catalog breadth & loyalty depth |
| `active_months` | Int64 | Distinct calendar months with purchases | Purchasing consistency |
| `cohort_month` | String | `YYYY-MM` of first purchase | Cohort assignment |
| `r_score`, `f_score`, `m_score` | Int (1-5) | Quintile bins for RFM components | Traditional baseline scoring |
| `rfm_segment` | String | Combined RFM tier (e.g., "Champions", "At Risk") | Rule-based baseline segmentation |

---

## 4. Probabilistic CLV & Next-Best-Action Output Table (`customer_clv_segments`)

| Field Name | Type | Definition & Source | Actionable Utility |
| :--- | :--- | :--- | :--- |
| `p_alive` | Float64 | $P(\text{Alive} \mid x, t_x, T)$ from BG/NBD | Probability that customer is currently active |
| `inactivity_prob_90d`| Float64 | Survival probability decay $1 - S(t + 90)$ | Likelihood of 90-day inactivity |
| `exp_purchases_90d` | Float64 | Expected repeat transactions in next 90 days ($E[Y(t)]$) | Demand forecasting |
| `exp_avg_monetary` | Float64 | Expected average spend per order from Gamma-Gamma | Spending potential |
| `clv_expected_90d` | Float64 | Probabilistic expected CLV (£) over 90 days | Customer valuation |
| `clv_lower_80pct` | Float64 | 80% Confidence Interval Lower Bound | Conservative value baseline |
| `clv_upper_80pct` | Float64 | 80% Confidence Interval Upper Bound | Optimistic potential ceiling |
| `action_segment` | String | Action-oriented segment (e.g., "High Value At Risk") | Operational grouping |
| `nba_recommended_action` | String | Next-Best-Action (e.g., `RETENTION`, `UPSELL`) | Prescriptive decision |
| `nba_priority` | String | `HIGH`, `MEDIUM`, `LOW` | Campaign execution rank |
| `nba_channel` | String | Recommended communication channel (Email, SMS, etc.) | Channel routing |
| `nba_reason` | String | Interpretable rationale for decision engine | Audit and marketer trust |
| `estimated_cost` | Float64 | Contact cost (£) for the recommended action | Budgeting & ROI modeling |
