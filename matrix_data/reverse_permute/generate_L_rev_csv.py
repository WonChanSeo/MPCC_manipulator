# -*- coding: utf-8 -*-
"""
Generate L_rev.csv (non-zero pattern in dense format)
Same format as L_rcm.csv
"""
import numpy as np

def symbolic_cholesky_fillin(A):
    """Compute L pattern with fill-in"""
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.int32)

    # Diagonal
    for i in range(n):
        L[i, i] = 1

    # Copy lower triangular from A (symmetric)
    for i in range(n):
        for j in range(i):
            if A[i, j] != 0 or A[j, i] != 0:
                L[i, j] = 1

    # Symbolic factorization (fill-in)
    for k in range(n):
        rows = np.where(L[k+1:, k] != 0)[0] + k + 1
        for i in rows:
            for j in rows:
                if i > j:
                    L[i, j] = 1

    return L

# Load original A
print('Loading A.csv...')
A = np.loadtxt('../A.csv', delimiter=',', dtype=np.float64)
n = A.shape[0]
print(f'Matrix size: {n}x{n}')

# Reverse permutation
print('\nApplying reverse permutation...')
perm_rev = np.arange(n-1, -1, -1)
A_rev = A[perm_rev, :][:, perm_rev]

# Compute L_rev with fill-in
print('Computing symbolic factorization...')
L_rev = symbolic_cholesky_fillin(A_rev)

# Save as CSV (same format as L_rcm.csv)
print('Saving L_rev.csv...')
np.savetxt('L_rev.csv', L_rev, fmt='%d', delimiter=',')

print(f'\nDone! L_rev.csv saved.')
print(f'Matrix size: {n}x{n}')
print(f'Non-zeros: {np.count_nonzero(L_rev)}')
