# Security, Privacy & Threat Model - BDS-34 Capstone Project

**Project**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Context**: Retail Customer Intelligence and Prescriptive CRM  

---

## 1. Overview and Security Principles

Customer relationship data, purchase histories, and predictive valuations carry significant business value and privacy implications under regulations such as GDPR and CCPA. Even when utilizing open-source datasets (e.g., UCI Online Retail II), this system enforces enterprise-grade security and privacy principles:

1. **Principle of Least Privilege**: Access to customer transaction records, predictions, and campaign simulation tools should be role-gated.
2. **Data Minimization & Anonymization**: Customer IDs are treated as sensitive pseudonymous identifiers. No direct Personally Identifiable Information (PII) like names, credit card numbers, or physical street addresses is required or retained.
3. **Defense in Depth**: Schema validation gates untrusted inputs, strict temporal barriers prevent data leakage, and sanity bounds constrain downstream business logic.

---

## 2. Threat Analysis (STRIDE Model)

| Threat Category | Potential Attack / Vulnerability Scenario | Impact Severity | Mitigations Implemented in BDS-34 |
| :--- | :--- | :--- | :--- |
| **Spoofing / Unauthorized Access** | Unauthorized actor accesses customer valuation lists or exports segmented lists for competitive intelligence. | **High** | Role-based view separation; no external open endpoints by default; credentials managed exclusively through environment variables. |
| **Tampering / Malicious Input** | Adversarial or corrupted transaction CSV/Excel with negative prices, SQL injection strings, or NaN exploits injected into pipeline. | **Medium** | Strict input schema validation via `Pydantic` and typed pipelines; automatic sanitization of non-numeric, out-of-range, and malformed records. |
| **Repudiation / Audit Failure** | Marketer or model execution results modified without historical audit trail of cleaning or segmentation decisions. | **Low** | Full audit trail logged with row counts, dropped record reasons, and immutable run timestamps in `reports/data_quality_report.json`. |
| **Information Disclosure** | Plaintext exposure of customer purchase volumes, unit pricing negotiations, or internal customer IDs to unauthorized internal users. | **High** | Customer IDs can be hashed/pseudonymized (`anonymize_customer_id: true`); raw zip/excel files are gitignored and excluded from version control. |
| **Denial of Service (DoS)** | Billion-row CSV upload or unbounded bootstrap simulations causing out-of-memory (OOM) CPU starvation. | **Medium** | Configurable chunking, memory bounds, vectorized NumPy operations, and constrained bootstrap iteration caps ($N \le 200$). |
| **Elevation of Privilege** | Dashboard user overrides campaign budget constraints or sends unauthorized marketing vouchers directly. | **Medium** | The campaign simulator operates strictly in an isolated synthetic simulation sandbox; no live dispatch integration exists without external authorization. |

---

## 3. Data Leakage & Machine Learning Specific Risks

### 3.1 Temporal Data Leakage
- **Risk**: Future transactions slipping into training splits, inflating model accuracy and understating churn risk.
- **Remediation**: Explicit cutoff engine (`src/features/splitter.py`). All feature derivations, RFM scores, and model fittings use strictly observations before $T_{cutoff}$.

### 3.2 Model Misuse & False Precision
- **Risk**: Marketers treating point-estimate CLV predictions as guaranteed future revenue, causing over-budgeting.
- **Remediation**: The system mandates 80% bootstrap confidence intervals ($[CLV_{lower}, CLV_{upper}]$) and explicitly surfaces uncertainty metrics.

### 3.3 Synthetic Data Conflation
- **Risk**: Stakeholders confusing simulated campaign response uplifts with real customer observations.
- **Remediation**: All synthetic artifacts are segregated in `data/synthetic/` and explicitly branded with UI warnings on dashboard pages.

---

## 4. Secure Development Practices
- `.gitignore` strictly protects raw customer data, credentials, and virtual environment binaries.
- No hardcoded API keys or credentials in code or YAML configurations.
- Pydantic models validate every user-configurable threshold before application runtime.
