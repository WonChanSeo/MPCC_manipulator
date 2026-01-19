# -*- coding: utf-8 -*-
"""
Generate CSC format for:
1. A_rev: Reverse permuted A matrix (before factorization)
2. L_rev: L matrix after factorization (with fill-in)
"""
import numpy as np

def to_csc(matrix, lower_only=False):
    """Convert dense matrix to CSC format"""
    n = matrix.shape[0]
    col_ptr = [0]
    row_idx = []
    values = []

    for j in range(n):
        for i in range(n):
            if lower_only and i < j:
                continue
            if matrix[i, j] != 0:
                row_idx.append(i)
                values.append(matrix[i, j])
        col_ptr.append(len(row_idx))

    return col_ptr, row_idx, values

def symbolic_cholesky_fillin(A):
    """Compute L pattern with fill-in"""
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    # Diagonal
    for i in range(n):
        L[i, i] = 1.0

    # Copy lower triangular from A (symmetric)
    for i in range(n):
        for j in range(i):
            if A[i, j] != 0 or A[j, i] != 0:
                L[i, j] = 1.0

    # Symbolic factorization (fill-in)
    for k in range(n):
        rows = np.where(L[k+1:, k] != 0)[0] + k + 1
        for i in rows:
            for j in rows:
                if i > j:
                    L[i, j] = 1.0

    return L

def save_csc(col_ptr, row_idx, values, prefix):
    """Save CSC format to files"""
    np.savetxt(f'{prefix}_col_ptr.txt', col_ptr, fmt='%d')
    np.savetxt(f'{prefix}_row_idx.txt', row_idx, fmt='%d')
    if values is not None and len(values) > 0:
        if isinstance(values[0], (int, np.integer)):
            np.savetxt(f'{prefix}_values.txt', values, fmt='%d')
        else:
            np.savetxt(f'{prefix}_values.txt', values, fmt='%.6f')
    print(f'Saved {prefix}_col_ptr.txt, {prefix}_row_idx.txt, {prefix}_values.txt')
    print(f'  n={len(col_ptr)-1}, nnz={len(row_idx)}')

# Load original A
print('Loading A.csv...')
A = np.loadtxt('../A.csv', delimiter=',', dtype=np.float64)
n = A.shape[0]
print(f'Matrix size: {n}x{n}')

# Reverse permutation
print('\nApplying reverse permutation...')
perm_rev = np.arange(n-1, -1, -1)
A_rev = A[perm_rev, :][:, perm_rev]

# === 1. A_rev (before factorization) - lower triangular only ===
print('\n=== A_rev (before factorization, lower triangular) ===')
A_rev_lower = np.tril(A_rev)
col_ptr, row_idx, values = to_csc(A_rev_lower, lower_only=False)
save_csc(col_ptr, row_idx, values, 'A_rev_lower_csc')

# === 2. L_rev (after factorization, with fill-in) ===
print('\n=== L_rev (after factorization, with fill-in) ===')
print('Computing symbolic factorization...')
L_rev = symbolic_cholesky_fillin(A_rev)
col_ptr, row_idx, values = to_csc(L_rev, lower_only=False)
save_csc(col_ptr, row_idx, values, 'L_rev_csc')

# === Summary ===
print('\n=== Summary ===')
print(f'A_rev lower triangular nnz: {np.count_nonzero(A_rev_lower)}')
print(f'L_rev nnz (with fill-in): {np.count_nonzero(L_rev)}')
print(f'Fill-in count: {np.count_nonzero(L_rev) - np.count_nonzero(A_rev_lower)}')

# Save permutation for reference
np.savetxt('perm_reverse.txt', perm_rev, fmt='%d')
print('\nSaved perm_reverse.txt')
