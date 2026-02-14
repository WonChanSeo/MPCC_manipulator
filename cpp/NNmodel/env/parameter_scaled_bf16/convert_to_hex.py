import struct
import os

def float_to_hex(val):
    """Convert a float value to its IEEE 754 FP32 hex string (8 chars)."""
    return struct.pack('>f', val).hex().upper()

src_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(src_dir, "hex")
os.makedirs(out_dir, exist_ok=True)

files = [f"weight_{i}.txt" for i in range(5)] + [f"bias_{i}.txt" for i in range(5)]

for fname in files:
    src_path = os.path.join(src_dir, fname)
    out_name = fname.replace(".txt", "_hex.txt")
    out_path = os.path.join(out_dir, out_name)

    with open(src_path, 'r') as f:
        lines = f.readlines()

    with open(out_path, 'w') as f:
        for line in lines:
            vals = line.strip().split()
            hex_vals = [float_to_hex(float(v)) for v in vals]
            f.write(' '.join(hex_vals) + '\n')

    print(f"{fname} -> {out_name} ({len(lines)} lines)")

print(f"\nDone. Output in: {out_dir}")
