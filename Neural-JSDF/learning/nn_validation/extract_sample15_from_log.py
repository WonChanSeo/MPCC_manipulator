#!/usr/bin/env python3
"""Extract Sample 15 output from the C++ log file."""

# Read log file
with open('cpp_validation_log.txt', 'r') as f:
    lines = f.readlines()

# Find the output section (starts at line 18, 0-indexed: 17)
# Each row is an output neuron (9 total), each column is a sample (100 total)

sample_idx = 15
print(f"Extracting Sample {sample_idx} from C++ log:")
print("="*50)

# Lines 18-26 contain the 9 output rows (17-25 in 0-indexed)
for output_idx in range(9):
    line_idx = 17 + output_idx  # 17 is where outputs start
    values = lines[line_idx].split()
    value = float(values[sample_idx])
    print(f"Output[{output_idx}]: {value}")

# Also check what cpp_validation_output.txt has
print("\n" + "="*50)
print("Values from cpp_validation_output.txt (line 16):")
print("="*50)

with open('cpp_validation_output.txt', 'r') as f:
    for i, line in enumerate(f):
        if i == sample_idx:
            values = line.split()
            for j, v in enumerate(values):
                print(f"Output[{j}]: {float(v)}")
            break
