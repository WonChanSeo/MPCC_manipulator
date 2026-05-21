# Exponent Scaling Comparison

Baseline: `final_truncate_off_no_exp_scaling_emergency_alpha_1p5_rti_diag_max_iter_250`

Treatment: `final_truncate_off_round_to_even_exp_scaling_diag_safe_max_iter_250`

| Metric | Baseline | Exp scaling | Change |
|---|---:|---:|---:|
| Manipulability mean | 0.0789312 | 0.0791516 | +0.279% |
| Contour error mean | 0.000873828 | 0.00083031 | -4.980% |
| Contour error RMS | 0.00115808 | 0.00109108 | -5.785% |
| Self-collision distance mean | 0.117821 | 0.114512 | -2.808% |
| Self-collision distance min | 0.0181088 | 0.0162042 | -10.518% |
| Nearest environment distance mean | 0.204608 | 0.206977 | +1.158% |
| Nearest environment distance min | 0.0793723 | 0.0827946 | +4.312% |
| End-effector speed mean | 0.13304 | 0.132985 | -0.041% |
| ADMM iterations mean | 158.258 | 150.452 | -4.933% |
| ADMM iterations p95 | 250 | 250 | +0.000% |
| ADMM iterations max | 250 | 250 | +0.000% |
| Rho updates mean | 0.748387 | 0.934194 | +24.828% |
| Total time mean (ms) | 68.4486 | 81.9919 | +19.786% |
| Total time p95 (ms) | 76.0593 | 80.602 | +5.973% |
| Solve QP mean (ms) | 1.72798 | 9.13277 | +428.522% |
| Scaling time mean (ms) | 0.0672139 | 0.976944 | +1353.484% |
| Init solver mean (ms) | 0.285125 | 3.60268 | +1163.542% |
| Factorization mean (ms) | 0.0756043 | 0.450064 | +495.289% |
| Permutation mean (ms) | 0.035255 | 0.039168 | +11.099% |
