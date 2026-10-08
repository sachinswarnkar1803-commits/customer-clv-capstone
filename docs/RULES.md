# Engineering Rules & Guidelines

**Project Name**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Project Code**: BDS-34  
**Course / Programme**: T.Y. B.Sc. Data Science – Semester V  
**Project Creators / Team**:
- **Sachin Swarnkar** (Roll No. `TDDS028B`)
- **Shivam Yadav** (Roll No. `TDDS044B`)  
**Status**: Engineering Governance Standard  
**Version**: 1.0.0  

---

## 1. Core Principles & Philosophy

1. **Scientific Honesty & Claim Integrity**: Never represent synthetic what-if campaign projections as observed empirical results. Clearly label simulation tools as planning instruments.
2. **Methodological Rigor**: Prioritize statistical correctness, temporal isolation, and explicit uncertainty quantification over black-box complexity.
3. **Reproducibility First**: Every pipeline stage, evaluation metric, and visualization must execute deterministically from raw data to final report.

---

## 2. Temporal Isolation & Leakage Prevention Rules

- **RULE-TEMP-01**: **Strict Cutoff Boundaries**: All features, frequency counts ($x$), recency values ($t_x$), and monetary sums ($\bar{M}$) for training must be computed strictly using transactions where $\text{InvoiceDate} \le T_{split}$.
- **RULE-TEMP-02**: **No Target Peeking**: Future holdout window transactions ($\text{InvoiceDate} > T_{split}$) may only be accessed during final model evaluation.
- **RULE-TEMP-03**: **Right-Censored Survival Events**: Time-to-inactivity events must be constructed based on observation windows available at $T_{split}$. Active customers who have not experienced the event at $T_{split}$ must be coded as censored ($E=0$), not as dead.

---

## 3. Modeling & Statistical Standards

- **RULE-STAT-01**: **Uncertainty Quantification**: Point-estimate CLV values must be accompanied by 80% Monte Carlo predictive intervals $[CLV_{lower}, CLV_{upper}]$.
- **RULE-STAT-02**: **Independence Assumption Check**: Before fitting the Gamma-Gamma model, the Pearson correlation between customer frequency and average monetary value must be computed and verified ($|r| < 0.20$ or documented).
- **RULE-STAT-03**: **Convergence Verification**: MLE optimization routines for BG/NBD and Gamma-Gamma must assert positive parameter estimates ($r, \alpha, a, b > 0; p, q, v > 0$) and check convergence flags.
- **RULE-STAT-04**: **Baseline Comparison**: Every probabilistic evaluation must include comparative benchmarks against traditional RFM quintile baselines.

---

## 4. Code Quality & Software Engineering Rules

- **RULE-CODE-01**: **Python Standard**: All code must conform to PEP 8 standards, targeting Python 3.10+.
- **RULE-CODE-02**: **Explicit Typing & Docstrings**: All public functions and classes must include type annotations (`typing`) and Google/NumPy-style docstrings describing parameters, return values, and exceptions.
- **RULE-CODE-03**: **Defensive Data Handling**: Data frames must be validated for missing customer IDs, negative prices, and non-conforming date formats during ingestion.
- **RULE-CODE-04**: **Logging over Print**: Production and pipeline scripts must utilize Python’s standard `logging` library with configurable log levels (`INFO`, `DEBUG`, `ERROR`), avoiding stray `print()` calls in library code.

---

## 5. Security & Privacy Rules

- **RULE-SEC-01**: **Identifier Sanitization**: Customer IDs are treated as confidential analytical identifiers. No PII (names, addresses, phone numbers) may be committed.
- **RULE-SEC-02**: **No Hardcoded Secrets**: No API keys, passwords, or cloud credentials may exist in source code or version control.
- **RULE-SEC-03**: **Security Auditing**: CI workflows must execute static security analysis (`bandit`) and dependency vulnerability scans (`pip-audit`) on pull requests.
- **RULE-SEC-04**: **Explicit Auth Claims**: Because this is an analytical capstone, do not claim enterprise role-based authentication or SSO unless an authentication provider is actually implemented.

---

## 6. Testing & CI/CD Governance

- **RULE-TEST-01**: **Unit Testing**: Every module (`src/data`, `src/models`, `src/clv`, `src/segmentation`, `src/nba`) must have corresponding unit tests in `tests/unit/`.
- **RULE-TEST-02**: **Deterministic Seeds**: All tests involving random sampling or Monte Carlo simulation must specify a fixed seed (`seed=42`).
- **RULE-TEST-03**: **Regression Testing**: Key pipeline outputs (evaluation metrics, schema contracts) must have regression tests to detect unexpected performance drops.
- **RULE-TEST-04**: **Passing CI**: No code may be merged or pushed without passing all automated GitHub Actions checks (linting, tests, security scans).
