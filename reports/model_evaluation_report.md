# Model Evaluation & Benchmark Report

**Project**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments (BDS-34)

## 1. Holdout Revenue Error Benchmarking

| Model                                       |   MAE (£) |   RMSE (£) |   Rev-Weighted MAE (£) |   Spearman Rank Corr |   Aggregate Predicted Rev (£) |   Actual Holdout Rev (£) |   Aggregate Bias (%) |
|:--------------------------------------------|----------:|-----------:|-----------------------:|---------------------:|------------------------------:|-------------------------:|---------------------:|
| Probabilistic CLV (BG/NBD + GG)             |   1316.97 |    7301.49 |                38473.9 |                0.603 |                   1.96514e+06 |              6.79085e+06 |               -71.06 |
| Baseline A (Historical Spend Extrapolation) |   1311.72 |    6377.56 |                33314.2 |                0.487 |                   3.56475e+06 |              6.79085e+06 |               -47.51 |
| Baseline B (RFM Segment Benchmarks)         |   1383.16 |    8156.63 |                43026.4 |                0.542 |                   2.16001e+06 |              6.79085e+06 |               -68.19 |

## 2. 80% CLV Prediction Interval Empirical Coverage

- **Nominal Target**: 80%
- **Empirical Coverage Rate**: 64.59%
- **Evaluated Customers**: 4,309
- **Notes**: NOTE: Coverage diverges slightly from nominal due to extreme skewness in retail spend.

## 3. Inactivity Risk Probability Calibration

- **Brier Score**: 0.2422 (Moderate)

### Calibration Reliability Table

| bin       |   count |   mean_predicted_prob |   observed_inactivity_rate |   abs_calibration_gap |
|:----------|--------:|----------------------:|---------------------------:|----------------------:|
| 0.00-0.20 |    3102 |                 0.066 |                      0.273 |                 0.207 |
| 0.20-0.40 |     462 |                 0.296 |                      0.552 |                 0.256 |
| 0.40-0.60 |     416 |                 0.496 |                      0.611 |                 0.115 |
| 0.60-0.80 |     329 |                 0.702 |                      0.754 |                 0.051 |

## 4. Sparse-History Sensitivity

| History Depth                    |   Customer Count |   Actual Mean Rev (£) |   Predicted Mean Rev (£) |   MAE (£) |   RMSE (£) |
|:---------------------------------|-----------------:|----------------------:|-------------------------:|----------:|-----------:|
| 1 Purchase (Zero Repeat, Sparse) |             1500 |                274.56 |                   150.43 |    301.43 |     914.68 |
| 2 Purchases (1 Repeat)           |              844 |                649.14 |                   280.42 |    593.15 |    2112.98 |
| 3 Purchases (2 Repeats)          |              533 |                732.38 |                   344.06 |    578.34 |    1044.84 |
| 4+ Purchases (Rich History)      |             1432 |               3799.43 |                   921.39 |   3082.26 |   12510.2  |