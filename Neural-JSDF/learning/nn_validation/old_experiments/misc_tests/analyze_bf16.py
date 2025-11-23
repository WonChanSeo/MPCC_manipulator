import numpy as np

# Load BF16 test data
data = np.loadtxt('data_bf16.csv', delimiter=',')

N = data.shape[0]
input_all = data[:N, :10]
nn_dist_all = data[:N, 10:19]
mesh_dist_all = data[:N, 19:]

print("=" * 60)
print("BF16 INFERENCE ANALYSIS")
print("Model: sdf_256x5_mesh_50000.pt")
print("Precision: BFloat16")
print("=" * 60)
print(f"\n{N} samples loaded")

# Use all samples without filtering
nn_dist = nn_dist_all
mesh_dist = mesh_dist_all

n_links = 9

# Calculate errors per link
print("-" * 60)
print("Per-Link Error Analysis (BF16)")
print("-" * 60)
print(f"{'Link':<8} {'L1 Error (cm)':<15} {'Max Error (cm)':<15}")
print("-" * 60)

for i in range(n_links):
    err = np.abs(mesh_dist[:, i] - nn_dist[:, i])
    l1_err = err.mean()
    max_err = err.max()
    print(f"Link {i:<3} {l1_err:>10.2f}       {max_err:>10.2f}")

print("-" * 60)

# Overall statistics
all_errors = np.abs(mesh_dist - nn_dist)
overall_l1 = all_errors.mean()
overall_max = all_errors.max()

print(f"\nOverall L1 Error:  {overall_l1:.2f} cm")
print(f"Overall Max Error: {overall_max:.2f} cm")
print("=" * 60)
