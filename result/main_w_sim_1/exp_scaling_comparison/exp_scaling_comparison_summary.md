# Exponent Scaling Comparison

Baseline: `final_truncate_addertree_all`

Treatment: `final_truncate_addertree_all_exp_scaling`

| Metric | Baseline | Exp scaling | Change |
|---|---:|---:|---:|
| Manipulability mean | 0.0787678 | 0.078771 | +0.004% |
| Contour error mean | 0.000949997 | 0.000944055 | -0.626% |
| Contour error RMS | 0.0012207 | 0.00121499 | -0.468% |
| Self-collision distance mean | 0.118991 | 0.118952 | -0.033% |
| Self-collision distance min | 0.0184775 | 0.018473 | -0.025% |
| Nearest environment distance mean | 0.201332 | 0.201241 | -0.045% |
| Nearest environment distance min | 0.0759357 | 0.0759286 | -0.009% |
| End-effector speed mean | 0.133196 | 0.133149 | -0.035% |
| ADMM iterations mean | 199.806 | 182.074 | -8.875% |
| ADMM iterations p95 | 500 | 500 | +0.000% |
| ADMM iterations max | 500 | 500 | +0.000% |
| Rho updates mean | 0.71447 | 0.717054 | +0.362% |
| Total time mean (ms) | 76.4444 | 75.7101 | -0.961% |
| Total time p95 (ms) | 85.3798 | 84.3848 | -1.165% |
| Solve QP mean (ms) | 2.06097 | 1.95552 | -5.117% |
| Scaling time mean (ms) | 0.0616742 | 0.0844716 | +36.964% |
| Init solver mean (ms) | 0.295962 | 0.320795 | +8.390% |
| Factorization mean (ms) | 0.0756511 | 0.0783055 | +3.509% |
| Permutation mean (ms) | 0.038811 | 0.0387176 | -0.241% |
