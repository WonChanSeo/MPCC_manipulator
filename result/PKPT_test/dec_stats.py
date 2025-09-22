#!/usr/bin/env python3
# dec_stats.py
import sys
import math
import argparse

def numbers_from_stream(stream):
    """
    stream에서 숫자를 한 줄씩 읽어 float으로 yield.
    - 빈 줄/주석(#, %) 스킵
    - 여러 토큰이 있으면 첫 토큰만 사용
    """
    for line in stream:
        s = line.strip()
        if not s:
            continue
        if s[0] in ('#', '%'):
            continue
        # 첫 토큰만 사용(공백/탭 기준)
        tok = s.split()[0]
        try:
            yield float(tok)
        except ValueError:
            # 숫자 아닌 줄은 건너뜀 (필요하면 raise로 바꿔도 됨)
            continue

def compute_stats(iterable):
    """
    iterable의 숫자들에 대해 count, min, max, mean(Kahan) 반환
    """
    n = 0
    vmin = math.inf
    vmax = -math.inf
    # Kahan summation
    s = 0.0
    c = 0.0
    for v in iterable:
        n += 1
        if v < vmin: vmin = v
        if v > vmax: vmax = v
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    if n == 0:
        return 0, math.nan, math.nan, math.nan
    return n, vmin, vmax, s / n

def main():
    ap = argparse.ArgumentParser(description="Compute count/min/max/mean from a .dec file (one number per line).")
    ap.add_argument("path", help="Input file path or '-' for stdin")
    ap.add_argument("--prec", type=int, default=17, help="Output precision (default: 17)")
    args = ap.parse_args()

    if args.path == "-":
        it = numbers_from_stream(sys.stdin)
    else:
        try:
            f = open(args.path, "r", encoding="utf-8", errors="ignore")
        except OSError as e:
            print(f"Error: cannot open {args.path}: {e}", file=sys.stderr)
            sys.exit(1)
        with f:
            it = numbers_from_stream(f)
            n, vmin, vmax, mean = compute_stats(it)
            # 출력
            p = args.prec
            fmt = f"{{:.{p}g}}"
            if n == 0:
                print("count: 0\nmin  : NaN\nmax  : NaN\nmean : NaN")
            else:
                print(f"count: {n}")
                print(f"min  : {fmt.format(vmin)}")
                print(f"max  : {fmt.format(vmax)}")
                print(f"mean : {fmt.format(mean)}")
            return

    # stdin 경로인 경우
    n, vmin, vmax, mean = compute_stats(it)
    p = args.prec
    fmt = f"{{:.{p}g}}"
    if n == 0:
        print("count: 0\nmin  : NaN\nmax  : NaN\nmean : NaN")
    else:
        print(f"count: {n}")
        print(f"min  : {fmt.format(vmin)}")
        print(f"max  : {fmt.format(vmax)}")
        print(f"mean : {fmt.format(mean)}")

if __name__ == "__main__":
    main()
