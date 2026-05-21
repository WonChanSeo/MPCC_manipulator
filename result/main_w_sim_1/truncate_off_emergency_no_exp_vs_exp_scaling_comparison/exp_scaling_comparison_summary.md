# Exponent Scaling Comparison

Baseline: `final_truncate_off_no_exp_scaling_emergency_max_iter_250`

Treatment: `final_truncate_off_round_to_even_exp_scaling_max_iter_250`

| Metric | Baseline | Exp scaling | Change |
|---|---:|---:|---:|
| Manipulability mean | 0.078804 | 0.0791486 | +0.437% |
| Contour error mean | 0.000945752 | 0.000830764 | -12.158% |
| Contour error RMS | 0.00121688 | 0.00109142 | -10.310% |
| Self-collision distance mean | 0.118878 | 0.114444 | -3.730% |
| Self-collision distance min | 0.0183676 | 0.0162042 | -11.778% |
| Nearest environment distance mean | 0.201518 | 0.20702 | +2.730% |
| Nearest environment distance min | 0.0761118 | 0.0827946 | +8.780% |
| End-effector speed mean | 0.133084 | 0.133028 | -0.042% |
| ADMM iterations mean | 155.617 | 150.452 | -3.319% |
| ADMM iterations p95 | 250 | 250 | +0.000% |
| ADMM iterations max | 250 | 250 | +0.000% |
| Rho updates mean | 0.677211 | 0.933505 | +37.846% |
| Total time mean (ms) | 69.1217 | 81.7151 | +18.219% |
| Total time p95 (ms) | 76.5598 | 79.8959 | +4.357% |
| Solve QP mean (ms) | 1.71389 | 9.08576 | +430.126% |
| Scaling time mean (ms) | 0.0673335 | 0.972697 | +1344.594% |
| Init solver mean (ms) | 0.286125 | 3.57233 | +1148.520% |
| Factorization mean (ms) | 0.0757497 | 0.448911 | +492.623% |
| Permutation mean (ms) | 0.0374218 | 0.0399491 | +6.753% |
