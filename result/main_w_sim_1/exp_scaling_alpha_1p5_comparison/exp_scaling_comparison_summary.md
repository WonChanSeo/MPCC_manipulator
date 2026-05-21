# Exponent Scaling Comparison

Baseline: `final_truncate_off_no_exp_scaling_emergency_alpha_1p5_diag_max_iter_250`

Treatment: `final_truncate_off_round_to_even_exp_scaling_diag_safe_max_iter_250`

| Metric | Baseline | Exp scaling | Change |
|---|---:|---:|---:|
| Manipulability mean | 0.0787745 | 0.0791456 | +0.471% |
| Contour error mean | 0.00095113 | 0.000831213 | -12.608% |
| Contour error RMS | 0.00122461 | 0.00109177 | -10.847% |
| Self-collision distance mean | 0.118794 | 0.114376 | -3.719% |
| Self-collision distance min | 0.0183276 | 0.0162042 | -11.586% |
| Nearest environment distance mean | 0.201437 | 0.207063 | +2.793% |
| Nearest environment distance min | 0.0759551 | 0.0827946 | +9.005% |
| End-effector speed mean | 0.133178 | 0.133069 | -0.082% |
| ADMM iterations mean | 158.075 | 150.452 | -4.822% |
| ADMM iterations p95 | 250 | 250 | +0.000% |
| ADMM iterations max | 250 | 250 | +0.000% |
| Rho updates mean | 0.680879 | 0.932171 | +36.907% |
| Total time mean (ms) | 69.5272 | 81.9961 | +17.934% |
| Total time p95 (ms) | 77.2271 | 80.6028 | +4.371% |
| Solve QP mean (ms) | 1.63936 | 9.13135 | +457.008% |
| Scaling time mean (ms) | 0.0672536 | 0.978075 | +1354.309% |
| Init solver mean (ms) | 0.285896 | 3.60645 | +1161.456% |
| Factorization mean (ms) | 0.0758724 | 0.450084 | +493.212% |
| Permutation mean (ms) | 0.0351117 | 0.0391693 | +11.556% |
