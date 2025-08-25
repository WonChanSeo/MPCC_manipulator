#!/usr/bin/env python3
import argparse, os, re
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def read_int_file(path: str, base: str = "auto") -> np.ndarray:
    """
    Read one integer per line for the whole file using a consistent base.
    base: 'hex' | 'dec' | 'auto'
    'auto': if filename ends with .hex OR any line has [a-fA-F] or 0x, treat the WHOLE file as hex.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # decide base once per file
    if base == "auto":
        if path.lower().endswith(".hex"):
            base = "hex"
        else:
            has_hex = False
            with open(path, "r") as f:
                for raw in f:
                    s = raw.split("#", 1)[0].strip()
                    if not s: 
                        continue
                    if s.lower().startswith("0x") or re.search(r"[a-fA-F]", s):
                        has_hex = True
                        break
            base = "hex" if has_hex else "dec"
    radix = 16 if base == "hex" else 10

    vals: List[int] = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.split("#", 1)[0].strip()
            if not s:
                continue
            if s.lower().startswith("0x"):
                vals.append(int(s, 16))
            else:
                vals.append(int(s, radix))
    return np.asarray(vals, dtype=np.int64)

def monotone_prefix_for_Ap(Ap: np.ndarray) -> int:
    if len(Ap) < 2 or Ap[0] < 0:
        return 0
    t = 0
    for j in range(len(Ap) - 1):
        p0, p1 = int(Ap[j]), int(Ap[j + 1])
        if p0 < 0 or p1 < p0:
            break
        t = j + 1
    return t  # equals n if well-formed

def build_coords_from_csc(Ap: np.ndarray, Ai: np.ndarray, ncols: int, nrows_hint: int = None
                          ) -> Tuple[np.ndarray, np.ndarray, int]:
    rows: List[int] = []
    cols: List[int] = []
    nnz_used = 0

    if nrows_hint is None:
        if len(Ai) == 0:
            nrows_max = ncols
        else:
            nrows_max = max(int(Ai.max()) + 1, ncols)
            nrows_max = max(nrows_max, 1)
    else:
        nrows_max = int(nrows_hint)

    for j in range(ncols):
        p0 = int(Ap[j]); p1 = int(Ap[j + 1])
        if p0 < 0 or p1 < p0:
            break
        p1 = min(p1, len(Ai))
        if p0 >= p1:
            continue
        col_rows = Ai[p0:p1]
        mask = (col_rows >= 0) & (col_rows < nrows_max)
        col_rows = col_rows[mask]
        if len(col_rows):
            rows.extend(col_rows.tolist())
            cols.extend([j] * len(col_rows))
            nnz_used += len(col_rows)

    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), nnz_used

def main():
    ap = argparse.ArgumentParser(description="Plot sparsity from CSC (Ap, Ai).")
    ap.add_argument("--ap", required=True, help="Path to Ap dump (hex or decimal, one integer per line).")
    ap.add_argument("--ai", required=True, help="Path to Ai dump (hex or decimal, one integer per line).")
    ap.add_argument("--out", default="sparsity.png", help="Output PNG path.")
    ap.add_argument("--rows", type=int, default=None, help="Matrix rows (if rectangular).")
    ap.add_argument("--show", action="store_true", help="Show interactive window after saving.")
    ap.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    ap.add_argument("--limit-nnz", type=int, default=None, help="Subsample to at most K points.")
    ap.add_argument("--base", choices=["hex","dec","auto"], default="auto",
                    help="Number base for the whole file (default: auto).")
    args = ap.parse_args()

    Ap = read_int_file(args.ap, base=args.base)
    Ai = read_int_file(args.ai, base=args.base)
    print(f"Parsed Ap/Ai as {args.base}.")

    if len(Ap) < 2:
        raise ValueError("Ap must have length >= 2 (n+1).")

    t = monotone_prefix_for_Ap(Ap)
    if t == 0:
        raise ValueError("Ap is malformed (non-monotone at the beginning).")
    if t < len(Ap) - 1:
        print(f"[warn] Ap non-monotone at index {t}. Using first {t} columns.")

    ncols = t
    rows, cols, nnz_used = build_coords_from_csc(Ap, Ai, ncols, args.rows)

    if args.limit_nnz is not None and nnz_used > args.limit_nnz:
        idx = np.random.RandomState(0).choice(nnz_used, size=args.limit_nnz, replace=False)
        rows = rows[idx]; cols = cols[idx]

    print(f"Ap length = {len(Ap)}  -> usable columns n = {ncols}")
    print(f"Ai length = {len(Ai)}")
    if nnz_used:
        print(f"nnz used = {nnz_used}")
        print(f"row range = [{rows.min()}, {rows.max()}], col range = [0, {ncols-1}]")
    else:
        print("No nonzeros to plot from the validated prefix.")

    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, s=2, marker='o')
    plt.gca().invert_yaxis()
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.title("Sparsity pattern (non-zero entries)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
