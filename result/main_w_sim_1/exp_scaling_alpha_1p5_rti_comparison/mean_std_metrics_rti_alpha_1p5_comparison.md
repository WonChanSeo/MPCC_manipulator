# Mean / Standard Deviation Comparison

Baseline: `final_truncate_off_no_exp_scaling_emergency_alpha_1p5_rti_diag_max_iter_250`

Exp scaling: `final_truncate_off_round_to_even_exp_scaling_diag_safe_max_iter_250`

Common samples: 1550

| Metric | Baseline mean +- std | Exp scaling mean +- std | Mean change | Std change |
|---|---:|---:|---:|---:|
| Manipulability | 0.0789312 +- 0.00955174 | 0.0791516 +- 0.00954671 | +0.279% | -0.053% |
| Contour error | 0.000873828 +- 0.000759979 m | 0.00083031 +- 0.000707839 m | -4.980% | -6.861% |
| Self-collision distance | 0.117821 +- 0.0647821 | 0.114512 +- 0.0668559 | -2.808% | +3.201% |
| Nearest environment distance | 0.204608 +- 0.110231 | 0.206977 +- 0.109387 | +1.158% | -0.766% |
| End-effector speed | 0.13304 +- 0.0809892 m/s | 0.132985 +- 0.0778985 m/s | -0.041% | -3.816% |
| ADMM iterations | 158.258 +- 72.0184 | 150.452 +- 68.7899 | -4.933% | -4.483% |
| Rho updates | 0.748387 +- 0.706592 | 0.934194 +- 0.710878 | +24.828% | +0.607% |
| Total time | 68.4486 +- 3.00758 ms | 81.9919 +- 41.1718 ms | +19.786% | +1268.936% |
| Solve QP time | 1.72798 +- 0.965678 ms | 9.13277 +- 13.8529 ms | +428.522% | +1334.521% |
| Scaling time | 0.0672139 +- 0.00235009 ms | 0.976944 +- 6.1344 ms | +1353.484% | +260928.447% |
