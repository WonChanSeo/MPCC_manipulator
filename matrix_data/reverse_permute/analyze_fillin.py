# -*- coding: utf-8 -*-
import numpy as np

A = np.loadtxt('A.csv', delimiter=',', dtype=np.float64)
n = A.shape[0]

# 각 column의 non-zero 개수 (lower triangular)
col_nnz = []
for j in range(n):
    nnz = 0
    for i in range(j, n):
        if A[i, j] != 0 or A[j, i] != 0:
            nnz += 1
    col_nnz.append(nnz)

col_nnz = np.array(col_nnz)

print('=== Column-wise nnz analysis ===')
print()

# 가장 nnz가 많은 column들
top_cols = np.argsort(col_nnz)[::-1][:15]
print('Top 15 columns with most non-zeros (lower triangular):')
for rank, col in enumerate(top_cols):
    print('  Rank {:2d}: Column {:3d} has {:3d} nnz'.format(rank+1, col, col_nnz[col]))

print()
print('Column 170-210 range:')
for col in range(170, 210):
    bar = '*' * min(col_nnz[col], 60)
    print('  Col {:3d}: {:3d} |{}'.format(col, col_nnz[col], bar))

# 어떤 row들과 연결되어 있는지 분석
print()
print('=== Analysis of high-nnz columns ===')
for col in top_cols[:5]:
    rows = []
    for i in range(col, n):
        if A[i, col] != 0 or A[col, i] != 0:
            rows.append(i)
    print('Column {} connects to rows: {}...'.format(col, rows[:10]))
