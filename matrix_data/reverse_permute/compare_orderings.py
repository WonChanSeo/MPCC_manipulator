# -*- coding: utf-8 -*-
import numpy as np

def symbolic_cholesky_fillin(A):
    """Compute L pattern with fill-in"""
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

def reverse_cuthill_mckee(A):
    """Simple RCM implementation"""
    n = A.shape[0]
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (A[i,j] != 0 or A[j,i] != 0):
                adj[i].append(j)

    # Sort by degree
    degrees = [len(adj[i]) for i in range(n)]

    # BFS from minimum degree node
    visited = [False] * n
    result = []

    while len(result) < n:
        # Find unvisited node with minimum degree
        min_deg = float('inf')
        start = -1
        for i in range(n):
            if not visited[i] and degrees[i] < min_deg:
                min_deg = degrees[i]
                start = i

        # BFS
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            result.append(node)
            # Sort neighbors by degree
            neighbors = sorted(adj[node], key=lambda x: degrees[x])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    # Reverse
    return result[::-1]

def block_ordering_mpc(n, n_vars_per_step, n_steps):
    """
    MPC block ordering: group by time step
    Assumes structure: [vars for all steps] [constraints for all steps]
    """
    # This is a simplified version - actual implementation depends on KKT structure
    perm = list(range(n))
    return perm

# Load matrix
print("Loading A.csv...")
A = np.loadtxt('A.csv', delimiter=',', dtype=np.float64)
n = A.shape[0]
print("Matrix size: {}x{}".format(n, n))

# 1. No permutation
print("\n=== 1. No Permutation ===")
L_none = symbolic_cholesky_fillin(A)
nnz_none = np.count_nonzero(L_none)
print("L nnz: {}".format(nnz_none))

# 2. RCM ordering
print("\n=== 2. Reverse Cuthill-McKee ===")
perm_rcm = reverse_cuthill_mckee(A)
perm_rcm = np.array(perm_rcm)
A_rcm = A[perm_rcm, :][:, perm_rcm]
L_rcm = symbolic_cholesky_fillin(A_rcm)
nnz_rcm = np.count_nonzero(L_rcm)
print("L nnz: {}".format(nnz_rcm))

# Check index regularity
perm_diff = np.diff(perm_rcm)
print("Permutation jumps > 10: {}".format(np.sum(np.abs(perm_diff) > 10)))
print("Max permutation jump: {}".format(np.max(np.abs(perm_diff))))

# 3. Load P_ref (AMD)
print("\n=== 3. AMD (P_ref) ===")
perm_amd = []
with open('../reference_matrix/P_ref.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            perm_amd.append(int(line))
perm_amd = np.array(perm_amd)
A_amd = A[perm_amd, :][:, perm_amd]
L_amd = symbolic_cholesky_fillin(A_amd)
nnz_amd = np.count_nonzero(L_amd)
print("L nnz: {}".format(nnz_amd))

perm_diff_amd = np.diff(perm_amd)
print("Permutation jumps > 10: {}".format(np.sum(np.abs(perm_diff_amd) > 10)))
print("Max permutation jump: {}".format(np.max(np.abs(perm_diff_amd))))

# Summary
print("\n=== Summary ===")
print("Method          | L nnz   | Fill ratio | Index regularity")
print("----------------|---------|------------|------------------")
print("No permutation  | {:6d}  | {:.2f}x      | Perfect (sequential)".format(nnz_none, nnz_none/nnz_amd))
print("RCM             | {:6d}  | {:.2f}x      | Mostly sequential".format(nnz_rcm, nnz_rcm/nnz_amd))
print("AMD (P_ref)     | {:6d}  | {:.2f}x      | Random jumps".format(nnz_amd, 1.0))

# Save RCM permutation and L pattern
np.savetxt('perm_rcm.txt', perm_rcm, fmt='%d')
np.savetxt('L_rcm.csv', L_rcm, delimiter=',', fmt='%d')
print("\nSaved perm_rcm.txt and L_rcm.csv")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].spy(L_none, markersize=0.3)
axes[0].set_title('No permutation\nnnz={}'.format(nnz_none))

axes[1].spy(L_rcm, markersize=0.3)
axes[1].set_title('RCM\nnnz={}'.format(nnz_rcm))

axes[2].spy(L_amd, markersize=0.3)
axes[2].set_title('AMD (P_ref)\nnnz={}'.format(nnz_amd))

plt.tight_layout()
plt.savefig('ordering_comparison.png', dpi=150)
print("Saved ordering_comparison.png")
