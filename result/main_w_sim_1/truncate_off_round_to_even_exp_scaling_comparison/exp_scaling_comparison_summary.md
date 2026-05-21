# Exponent Scaling Comparison

Baseline: `final_truncate_addertree_all_max_iter_250`

Treatment: `final_truncate_off_round_to_even_exp_scaling_max_iter_250`

| Metric | Baseline | Exp scaling | Change |
|---|---:|---:|---:|
| Manipulability mean | 0.0787929 | 0.0791486 | +0.451% |
| Contour error mean | 0.000941329 | 0.000830764 | -11.746% |
| Contour error RMS | 0.00121755 | 0.00109142 | -10.359% |
| Self-collision distance mean | 0.118874 | 0.114444 | -3.726% |
| Self-collision distance min | 0.0184362 | 0.0162042 | -12.107% |
| Nearest environment distance mean | 0.201665 | 0.20702 | +2.656% |
| Nearest environment distance min | 0.0763557 | 0.0827946 | +8.433% |
| End-effector speed mean | 0.133075 | 0.133028 | -0.035% |
| ADMM iterations mean | 158.425 | 150.452 | -5.033% |
| ADMM iterations p95 | 250 | 250 | +0.000% |
| ADMM iterations max | 250 | 250 | +0.000% |
| Rho updates mean | 0.690123 | 0.933505 | +35.267% |
| Total time mean (ms) | 76.0594 | 81.7151 | +7.436% |
| Total time p95 (ms) | 84.1355 | 79.8959 | -5.039% |
| Solve QP mean (ms) | 1.65779 | 9.08576 | +448.065% |
| Scaling time mean (ms) | 0.0680223 | 0.972697 | +1329.968% |
| Init solver mean (ms) | 0.290685 | 3.57233 | +1128.938% |
| Factorization mean (ms) | 0.0770821 | 0.448911 | +482.380% |
| Permutation mean (ms) | 0.0358283 | 0.0399491 | +11.501% |
