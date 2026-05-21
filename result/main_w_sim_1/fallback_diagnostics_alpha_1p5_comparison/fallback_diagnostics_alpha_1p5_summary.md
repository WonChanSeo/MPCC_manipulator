# Fallback Diagnostics: Alpha 1.5 Check

| Run | Samples | Cold starts | Previous-invalid fallbacks | Final solve failures | Final max-iter failures | Iter>=250 solved | Internal status=7 warnings | Path deviation fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline alpha=1.6 | 1549 | 1 | 0 | 0 | 0 | 433 | 55 | 0 |
| Baseline alpha=1.5 | 1548 | 1 | 0 | 0 | 0 | 484 | 66 | 0 |
| Exp scaling alpha=1.5 | 1550 | 78 | 77 | 77 | 77 | 272 | 77 | 0 |

## Status Counts

- Baseline alpha=1.6: SOLVED=1549
- Baseline alpha=1.5: SOLVED=1548
- Exp scaling alpha=1.5: SOLVED=1473, QP_MaxIterReached=77

## Exp Failure Clusters

- Final max-iter/failure clusters: 229 (1), 239 (1), 259 (1), 262 (1), 265 (1), 271 (1), 274 (1), 279 (1), 282 (1), 286 (1), 289 (1), 291 (1), 293 (1), 296 (1), 299 (1), 321 (1), 324 (1), 328 (1), 332 (1), 335 (1), 338 (1), 341 (1), 804 (1), 811 (1), 825 (1), 836 (1), 838 (1), 846 (1), 852 (1), 856 (1), 861 (1), 866 (1), 870 (1), 874 (1), 878 (1), 882 (1), 886 (1), 900 (1), 902 (1), 904 (1), 906 (1), 917 (1), 923 (1), 926 (1), 929 (1), 932 (1), 940 (1), 954 (1), 956 (1), 958 (1), 972 (1), 974 (1), 976 (1), 978 (1), 1000 (1), 1003 (1), 1008 (1), 1012 (1), 1021 (1), 1023 (1), 1027 (1), 1029 (1), 1040 (1), 1045 (1), 1081 (1), 1085 (1), 1091 (1), 1096 (1), 1108 (1), 1117 (1), 1127 (1), 1142 (1), 1146 (1), 1165 (1), 1235 (1), 1294 (1), 1314 (1)
- Previous-invalid fallback clusters: 230 (1), 240 (1), 260 (1), 263 (1), 266 (1), 272 (1), 275 (1), 280 (1), 283 (1), 287 (1), 290 (1), 292 (1), 294 (1), 297 (1), 300 (1), 322 (1), 325 (1), 329 (1), 333 (1), 336 (1), 339 (1), 342 (1), 805 (1), 812 (1), 826 (1), 837 (1), 839 (1), 847 (1), 853 (1), 857 (1), 862 (1), 867 (1), 871 (1), 875 (1), 879 (1), 883 (1), 887 (1), 901 (1), 903 (1), 905 (1), 907 (1), 918 (1), 924 (1), 927 (1), 930 (1), 933 (1), 941 (1), 955 (1), 957 (1), 959 (1), 973 (1), 975 (1), 977 (1), 979 (1), 1001 (1), 1004 (1), 1009 (1), 1013 (1), 1022 (1), 1024 (1), 1028 (1), 1030 (1), 1041 (1), 1046 (1), 1082 (1), 1086 (1), 1092 (1), 1097 (1), 1109 (1), 1118 (1), 1128 (1), 1143 (1), 1147 (1), 1166 (1), 1236 (1), 1295 (1), 1315 (1)

## Notes

- `Internal status=7 warnings` are console-level solveQP warnings and do not necessarily become MPC-level fallback calls.
- Fallback-call evidence should be read from `cold_start`/`reason=previous_invalid_guess` in `fallback_diagnostics.csv`.
- `status=5` in the MPC diagnostic CSV maps to `QP_MaxIterReached`.
