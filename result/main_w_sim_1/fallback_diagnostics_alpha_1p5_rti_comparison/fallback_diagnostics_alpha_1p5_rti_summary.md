# Fallback Diagnostics: RTI, alpha=1.5

Both runs use truncate off, max_iter=250, RTI=true, alpha=1.5. ASIC testcase outputs were not overwritten; logs were routed under each result directory.

| Run | Samples | Previous-invalid fallback | Final max-iter failure | iter>=250 solved | Path-deviation fallback | RTI fail console |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RTI alpha=1.5 | 1550 | 79 (5.10%) | 79 (5.10%) | 377 | 0 | 79 |
| Exp scaling RTI alpha=1.5 | 1550 | 77 (4.97%) | 77 (4.97%) | 272 | 0 | 77 |

Delta exp-baseline previous-invalid fallback: -2

Max-iter overlap: 15 shared solve_count values. Baseline-only: 64, exp-only: 62.

Shared max-iter solve_counts: [825, 861, 886, 902, 906, 917, 940, 954, 1023, 1027, 1029, 1096, 1108, 1127, 1235]

Baseline-only max-iter solve_counts: [219, 224, 226, 228, 230, 241, 302, 305, 760, 773, 776, 779, 791, 798, 805, 821, 830, 842, 851, 853, 855, 867, 871, 891, 897, 908, 919, 921, 934, 948, 952, 980, 981, 983, 984, 987, 992, 1002, 1005, 1013, 1026, 1036, 1046, 1047, 1049, 1053, 1073, 1074, 1075, 1076, 1078, 1080, 1082, 1088, 1093, 1101, 1121, 1131, 1139, 1147, 1174, 1203, 1207, 1288]

Exp-only max-iter solve_counts: [229, 239, 259, 262, 265, 271, 274, 279, 282, 286, 289, 291, 293, 296, 299, 321, 324, 328, 332, 335, 338, 341, 804, 811, 836, 838, 846, 852, 856, 866, 870, 874, 878, 882, 900, 904, 923, 926, 929, 932, 956, 958, 972, 974, 976, 978, 1000, 1003, 1008, 1012, 1021, 1040, 1045, 1081, 1085, 1091, 1117, 1142, 1146, 1165, 1294, 1314]
