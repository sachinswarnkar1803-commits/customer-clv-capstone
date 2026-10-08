# Industry Problem Brief — BDS-34

## 1. Business problem

Retail teams need to decide which customers should receive retention, reactivation, upsell, cross-sell, or low-cost engagement actions. Historical revenue alone does not distinguish customer purchase frequency, probability of remaining active, expected future value, and uncertainty.

This project turns transaction history into a customer-level decision layer using probabilistic purchase modelling, monetary-value modelling, inactivity survival analysis, cohort dynamics, and a transparent Next-Best-Action engine.

## 2. Stakeholders

| Stakeholder | Need | Decision supported |
|---|---|---|
| CRM / retention manager | Identify valuable customers at risk | Retention and win-back targeting |
| Marketing analyst | Estimate future customer value | Budget allocation and audience sizing |
| Campaign manager | Select action and channel | Next-Best-Action execution |
| Data scientist | Validate model quality | Holdout evaluation and calibration |
| Engineering / platform owner | Reproduce and monitor the system | Pipeline, CI, deployment and drift monitoring |
| Academic reviewer | Verify evidence and methodology | Reproducibility and assessment |

## 3. Current process and pain points

A basic retail workflow often uses total historical spend or manually defined RFM rules. These approaches can be useful baselines but do not explicitly model repeat-purchase probability, monetary uncertainty, or time-to-inactivity.

The capstone therefore treats RFM and historical-spend methods as benchmarks rather than replacing them with an unsupported claim of universal superiority.

## 4. User stories

- As a retention manager, I want to identify high-value customers with elevated inactivity risk so that I can review an appropriate retention action.
- As a marketing analyst, I want an expected CLV and uncertainty interval so that I can distinguish expected value from prediction risk.
- As a campaign manager, I want a recommended action, channel, cost, and rationale for each customer.
- As a data scientist, I want a temporal holdout and baseline comparison so that future information does not leak into training.
- As an engineering reviewer, I want tests, dependency controls, security scans, and reproducible startup instructions.

## 5. Functional requirements

1. Ingest and validate transaction data.
2. Build leakage-safe customer features.
3. Produce cohort retention and revenue dynamics.
4. Fit BG/NBD purchase and Gamma-Gamma monetary models.
5. Estimate time-to-inactivity with right-censoring.
6. Compute CLV across configurable horizons and predictive uncertainty intervals.
7. Generate action-oriented segments and Next-Best-Actions.
8. Simulate synthetic campaign scenarios.
9. Evaluate holdout revenue, interval coverage, inactivity calibration, sparse-history sensitivity, and segment transitions.
10. Monitor feature and segment drift.
11. Provide an interactive dashboard and exportable action queue.

## 6. Non-functional requirements

- Deterministic random seed for reproducible simulations.
- Exact dependency versions in `requirements.txt`.
- Automated unit/integration tests in CI.
- Dependency vulnerability and static security scanning.
- No production authentication claim unless an authentication layer is actually implemented.
- Customer identifiers should be treated as confidential analytical identifiers; this capstone does not implement a production identity-access system.

## 7. Misuse and abuse cases

- Treating synthetic campaign response assumptions as observed business outcomes.
- Using model outputs as guaranteed individual revenue.
- Interpreting a prediction interval as a confidence interval for fitted parameters.
- Using a customer-level recommendation without checking business policy, consent, frequency caps, and channel eligibility.
- Uploading uncontrolled datasets that can cause excessive memory or compute use.

## 8. Scope

### In scope

Transaction analytics, probabilistic CLV, cohort dynamics, inactivity risk, action segmentation, Next-Best-Action logic, synthetic campaign simulation, evaluation, monitoring, testing, Docker packaging, and Streamlit decision support.

### Out of scope

Real campaign execution, payment processing, customer identity resolution, production authentication/RBAC, personally identifiable customer enrichment, and causal measurement of campaign uplift from real randomized experiments.

## 9. Success metrics

| Area | Metric | Evidence |
|---|---|---|
| Forecasting | MAE, RMSE, Spearman rank correlation | Holdout report |
| Uncertainty | Empirical interval coverage | Evaluation report |
| Risk modelling | Brier score and reliability table | Calibration report |
| Segmentation | Customer transition matrix and agreement rate | Segment stability report |
| Reliability | PSI and monitoring status | Monitoring report |
| Engineering | Test suite, CI, security scan | GitHub Actions |
| Reproducibility | Clean environment setup and deterministic pipeline | README / Docker |

## 10. Prioritized backlog

| Priority | Item | Acceptance criteria |
|---|---|---|
| P0 | Correct inactivity target | Event means reaching configured inactivity threshold; censoring is explicit |
| P0 | Correct uncertainty terminology | Documentation calls intervals Monte Carlo prediction intervals |
| P0 | Segment stability | Transition matrix and agreement are generated and displayed |
| P0 | Reproducible deployment | Docker starts with artifacts or clearly enters setup mode |
| P1 | Security scanning | CI runs `pip-audit` and `bandit` |
| P1 | Professional dashboard | No decorative emoji; interactive filters and exports work |
| P1 | Performance evidence | Benchmark script produces a machine-readable report |
| P2 | Production API | Explicitly out of scope for this capstone unless separately implemented |
