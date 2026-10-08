# Contribution Evidence Log — BDS-34

**Project**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  
**Team**: Sachin Swarnkar (Roll No. `TDDS028B`) and Shivam Yadav (Roll No. `TDDS044B`)  

## Evidence policy

This document records deliverables that can be supported by repository evidence. It does **not** manufacture commit, pull-request, review, or pair-programming history. The supplied project snapshot contained one initial Git commit, so future contribution claims must be backed by actual Git commits, branches, pull requests, or review records.

## Verified project areas

| Area | Repository evidence |
|---|---|
| Data ingestion and validation | `src/data/`, `src/validation/`, validation reports |
| Temporal feature engineering | `src/features/` |
| Cohort dynamics | `src/cohort/` and cohort artifacts |
| BG/NBD and Gamma-Gamma | `src/models/purchase_model.py`, `src/models/monetary_model.py` |
| Time-to-inactivity survival | `src/models/inactivity_model.py` |
| CLV and uncertainty | `src/clv/clv_calculator.py` |
| Action segmentation | `src/segmentation/segmenter.py` |
| Next-Best-Action | `src/nba/nba_engine.py` |
| Scenario simulation | `src/simulation/campaign_simulator.py` |
| Evaluation and monitoring | `src/evaluation/`, `src/monitoring/` |
| Interactive application | `dashboard/app.py` |
| Tests and CI | `tests/`, `.github/workflows/ci.yml` |
| Deployment | `Dockerfile`, `docker-compose.yml`, `scripts/docker_entrypoint.sh` |
| Requirements analysis | `docs/problem_brief.md` |
| Solution design | `docs/solution_design_pack.md`, `docs/architecture/` |

## Remediation evidence to commit as a team

Create genuine commits and code reviews for the changes below. Use the real student names and dates from the team's actual Git history when completing the final submission:

1. Correct time-to-inactivity event construction.
2. Rename and document Monte Carlo predictive intervals.
3. Add segment transition/stability evaluation.
4. Align the 90-day inactivity calibration target with the prediction horizon.
5. Add reproducible Docker startup behaviour.
6. Add security/dependency scanning to CI.
7. Add performance evidence.
8. Add the industry problem brief and solution design pack.
9. Replace the dashboard with the professional no-emoji interface.

## Final submission requirement

The team should keep the contribution log synchronized with actual Git evidence. Do not claim a pull request, review, pair-programming session, or commit unless it exists in the repository or the team's submission evidence.
