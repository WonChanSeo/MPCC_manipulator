# Exponent Scaling Comparison

Baseline: `final_truncate_addertree_all_max_iter_250`

Treatment: `final_truncate_addertree_all_exp_scaling_max_iter_250`

| Metric | Baseline | Exp scaling | Change |
|---|---:|---:|---:|
| Manipulability mean | 0.0898281 | 0.0898184 | -0.011% |
| Contour error mean | 9.31755e-05 | 0 | -100.000% |
| Contour error RMS | 0.000100099 | 0 | -100.000% |
| Self-collision distance mean | 0.216225 | 0.216248 | +0.011% |
| Self-collision distance min | 0.21621 | 0.216248 | +0.017% |
| Nearest environment distance mean | 0.147317 | 0.147165 | -0.103% |
| Nearest environment distance min | 0.147195 | 0.146824 | -0.252% |
| End-effector speed mean | 0.0284102 | 0 | -100.000% |
| ADMM iterations mean | 112.5 | 250 | +122.222% |
| ADMM iterations p95 | 142.5 | 250 | +75.439% |
| ADMM iterations max | 150 | 250 | +66.667% |
| Rho updates mean | 1 | 3 | +200.000% |
| Total time mean (ms) | 82.6234 | 201.756 | +144.187% |
| Total time p95 (ms) | 100.374 | 501.831 | +399.959% |
| Solve QP mean (ms) | 5.37098 | 55.1796 | +927.366% |
| Scaling time mean (ms) | 0.0664922 | 11.2683 | +16846.781% |
| Init solver mean (ms) | 0.295419 | 37.467 | +12582.629% |
| Factorization mean (ms) | 0.0759813 | 0.976379 | +1185.026% |
| Permutation mean (ms) | 0.0374057 | 0.0428422 | +14.534% |
