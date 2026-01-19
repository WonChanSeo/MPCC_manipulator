# -*- coding: utf-8 -*-
"""
MPC-aware block ordering for ASIC-friendly factorization
"""
import numpy as np
import matplotlib.pyplot as plt

def symbolic_cholesky_fillin(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        L[i, i] = 1
    for i in range(n):
        for j in range(i):
            if A[i, j] != 0 or A[j, i] != 0:
                L[i, j] = 1
    for k in range(n):
        rows = np.where(L[k+1:, k] != 0)[0] + k + 1
        for i in rows:
            for j in rows:
                if i > j:
                    L[i, j] = 1
    return L

def compute_bandwidth(L):
    """Compute the bandwidth of lower triangular matrix L"""
    n = L.shape[0]
    max_bw = 0
    for i in range(n):
        for j in range(i+1):
            if L[i, j] != 0:
                max_bw = max(max_bw, i - j)
    return max_bw

def analyze_block_structure(A, block_size=9):
    """Analyze block structure assuming block_size variables per time step"""
    n = A.shape[0]
    n_blocks = n // block_size

    print("Analyzing block structure (block_size={})...".format(block_size))

    # Check which blocks connect to which
    connections = []
    for bi in range(n_blocks):
        for bj in range(bi+1):
            block = A[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size]
            if np.any(block != 0):
                connections.append((bi, bj))

    print("Block connections (non-zero blocks):")
    print("  Diagonal blocks: {}".format(sum(1 for c in connections if c[0] == c[1])))
    print("  Off-diagonal blocks: {}".format(sum(1 for c in connections if c[0] != c[1])))

    # Find max block distance
    max_dist = max(c[0] - c[1] for c in connections)
    print("  Max block distance: {}".format(max_dist))

    return connections

# Load matrix
print("Loading A.csv...")
A = np.loadtxt('A.csv', delimiter=',', dtype=np.float64)
n = A.shape[0]
print("Matrix size: {}x{}".format(n, n))

# Analyze block structure with different block sizes
print("\n=== Block Structure Analysis ===")
for bs in [7, 8, 9, 11, 14]:
    if n % bs == 0 or True:
        analyze_block_structure(A, bs)
        print()

# Try different "ASIC-friendly" orderings

# 1. Identity (no permutation) - baseline
print("\n=== 1. No Permutation ===")
L_none = symbolic_cholesky_fillin(A)
bw_none = compute_bandwidth(L_none)
print("L nnz: {}, bandwidth: {}".format(np.count_nonzero(L_none), bw_none))

# 2. Reverse order (sometimes helps with arrow-head)
print("\n=== 2. Reverse Order ===")
perm_rev = np.arange(n-1, -1, -1)
A_rev = A[perm_rev, :][:, perm_rev]
L_rev = symbolic_cholesky_fillin(A_rev)
bw_rev = compute_bandwidth(L_rev)
print("L nnz: {}, bandwidth: {}".format(np.count_nonzero(L_rev), bw_rev))

# 3. Move coupling rows to end (manual arrow-head handling)
print("\n=== 3. Coupling to End ===")
# Find rows/cols with many connections (coupling nodes)
row_nnz = np.sum(A != 0, axis=1)
col_nnz = np.sum(A != 0, axis=0)
total_nnz = row_nnz + col_nnz

# Sort by nnz (low nnz first, high nnz last)
perm_coupling = np.argsort(total_nnz)
A_coupling = A[perm_coupling, :][:, perm_coupling]
L_coupling = symbolic_cholesky_fillin(A_coupling)
bw_coupling = compute_bandwidth(L_coupling)
print("L nnz: {}, bandwidth: {}".format(np.count_nonzero(L_coupling), bw_coupling))

# 4. Load RCM and AMD for comparison
print("\n=== 4. RCM (from previous) ===")
perm_rcm = np.loadtxt('perm_rcm.txt', dtype=int)
A_rcm = A[perm_rcm, :][:, perm_rcm]
L_rcm = symbolic_cholesky_fillin(A_rcm)
bw_rcm = compute_bandwidth(L_rcm)
print("L nnz: {}, bandwidth: {}".format(np.count_nonzero(L_rcm), bw_rcm))

print("\n=== 5. AMD (P_ref) ===")
perm_amd = []
with open('../reference_matrix/P_ref.txt', 'r') as f:
    for line in f:
        if line.strip():
            perm_amd.append(int(line.strip()))
perm_amd = np.array(perm_amd)
A_amd = A[perm_amd, :][:, perm_amd]
L_amd = symbolic_cholesky_fillin(A_amd)
bw_amd = compute_bandwidth(L_amd)
print("L nnz: {}, bandwidth: {}".format(np.count_nonzero(L_amd), bw_amd))

# Summary
print("\n" + "="*70)
print("=== Summary (ASIC perspective) ===")
print("="*70)
print("{:<20} | {:>8} | {:>10} | {:>15}".format("Method", "L nnz", "Bandwidth", "ASIC suitability"))
print("-"*70)
print("{:<20} | {:>8} | {:>10} | {:>15}".format("No permutation", np.count_nonzero(L_none), bw_none, "Bad (dense)"))
print("{:<20} | {:>8} | {:>10} | {:>15}".format("Reverse order", np.count_nonzero(L_rev), bw_rev, "?"))
print("{:<20} | {:>8} | {:>10} | {:>15}".format("Coupling to end", np.count_nonzero(L_coupling), bw_coupling, "Better"))
print("{:<20} | {:>8} | {:>10} | {:>15}".format("RCM", np.count_nonzero(L_rcm), bw_rcm, "Good (banded)"))
print("{:<20} | {:>8} | {:>10} | {:>15}".format("AMD (P_ref)", np.count_nonzero(L_amd), bw_amd, "Best nnz"))

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].spy(L_none, markersize=0.3)
axes[0,0].set_title('No perm: nnz={}, bw={}'.format(np.count_nonzero(L_none), bw_none))

axes[0,1].spy(L_rev, markersize=0.3)
axes[0,1].set_title('Reverse: nnz={}, bw={}'.format(np.count_nonzero(L_rev), bw_rev))

axes[0,2].spy(L_coupling, markersize=0.3)
axes[0,2].set_title('Coupling to end: nnz={}, bw={}'.format(np.count_nonzero(L_coupling), bw_coupling))

axes[1,0].spy(L_rcm, markersize=0.3)
axes[1,0].set_title('RCM: nnz={}, bw={}'.format(np.count_nonzero(L_rcm), bw_rcm))

axes[1,1].spy(L_amd, markersize=0.3)
axes[1,1].set_title('AMD: nnz={}, bw={}'.format(np.count_nonzero(L_amd), bw_amd))

# Permutation regularity visualization
axes[1,2].plot(perm_rcm, 'b-', alpha=0.5, label='RCM')
axes[1,2].plot(perm_amd, 'r-', alpha=0.5, label='AMD')
axes[1,2].plot(perm_coupling, 'g-', alpha=0.5, label='Coupling')
axes[1,2].set_title('Permutation index mapping')
axes[1,2].legend()
axes[1,2].set_xlabel('New index')
axes[1,2].set_ylabel('Original index')

plt.tight_layout()
plt.savefig('asic_ordering_comparison.png', dpi=150)
print("\nSaved asic_ordering_comparison.png")
