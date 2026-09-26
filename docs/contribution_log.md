# Capstone Contribution Log - BDS-34

**Programme**: T.Y. B.Sc. Data Science – Semester V  
**Project Code**: BDS-34  
**Project Title**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Team**: Student 1 (Sachin - Lead Engineer / Architect) & Student 2 (Co-Lead / Analyst)  

---

## Contribution Audit Trail

| Date | Student | Activity | Evidence & Deliverable | Hours |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-24 | Student 1 | Project architecture setup, repository scaffolding, configuration system design | `customer-clv-capstone/`, `configs/default_config.yaml`, `src/config/config.py` | 4.0 |
| 2026-09-24 | Student 2 | Requirements analysis, data dictionary compilation, threat modeling | `docs/data_dictionary.md`, `docs/threat_model.md`, `README.md` | 3.5 |
| 2026-09-24 | Student 1 | Ingestion pipeline implementation & UCI dataset retrieval module | `src/data/ingest.py`, `tests/unit/test_validation.py` | 4.0 |
| 2026-09-24 | Student 2 | Data validation engine, audit trail tracking, and schema test suites | `src/validation/validator.py`, data quality reporting | 4.0 |
| 2026-09-24 | Student 1 | Feature engineering pipeline, temporal split engine, RFM baseline | `src/features/`, `src/models/baseline.py` | 4.5 |
| 2026-09-24 | Student 2 | Cohort analysis module, retention matrix computations, and visualizations | `src/cohort/cohort_analysis.py` | 3.5 |
| 2026-09-24 | Student 1 | Probabilistic BG/NBD purchase model & Gamma-Gamma monetary model | `src/models/purchase_model.py`, `src/models/monetary_model.py` | 5.0 |
| 2026-09-24 | Student 2 | Customer inactivity survival modeling (Kaplan-Meier, Weibull hazards) | `src/models/inactivity_model.py` | 4.0 |
| 2026-09-24 | Student 1 | Probabilistic CLV engine with bootstrap uncertainty intervals | `src/clv/clv_calculator.py` | 4.5 |
| 2026-09-24 | Student 2 | Action-oriented customer segmentation and Next-Best-Action engine | `src/segmentation/`, `src/nba/` | 4.0 |
| 2026-09-24 | Student 1 | Interactive campaign scenario simulator and synthetic experimentation layer | `src/simulation/campaign_simulator.py` | 4.0 |
| 2026-09-24 | Student 2 | Holdout evaluation, calibration, coverage, and model monitoring | `src/evaluation/`, `src/monitoring/` | 4.5 |
| 2026-09-24 | Student 1 & 2 | Streamlit multi-page command dashboard implementation | `dashboard/app.py`, `dashboard/pages/` | 6.0 |
| 2026-09-24 | Student 1 & 2 | Integration testing, Docker containerization, and final capstone documentation | `tests/integration/`, `Dockerfile`, `docker-compose.yml` | 3.5 |

*Note: All work is developed iteratively in pair programming with code reviews, ensuring comprehensive mutual understanding of both the mathematical modeling and software engineering components.*
