# -*- coding: utf-8 -*-
"""
CSC column pointer 파일을 분석하는 스크립트
*_p.hex 또는 *p.txt 파일에서 열별 element 개수를 분석합니다.
"""
import sys

def load_col_ptr(filepath):
    """col_ptr 파일을 읽어옵니다."""
    values = []
    is_hex = filepath.endswith('.hex')

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_hex:
                values.append(int(line, 16))
            else:
                # 8자리 또는 16자리 고정 길이면 hex로 간주
                if len(line) in [8, 16] and all(c in '0123456789abcdefABCDEF' for c in line):
                    values.append(int(line, 16))
                else:
                    values.append(int(line))

    return values

def analyze_col_ptr(col_ptr):
    """col_ptr를 분석하여 통계를 반환합니다."""
    n_cols = len(col_ptr) - 1
    nnz = col_ptr[-1]

    col_counts = []
    empty_cols = 0
    empty_col_indices = []

    for i in range(n_cols):
        count = col_ptr[i + 1] - col_ptr[i]
        col_counts.append(count)
        if count == 0:
            empty_cols += 1
            empty_col_indices.append(i)

    max_count = max(col_counts) if col_counts else 0
    min_count = min(col_counts) if col_counts else 0
    avg_count = sum(col_counts) / len(col_counts) if col_counts else 0

    # 최대 개수를 가진 열 찾기
    max_cols = [i for i, c in enumerate(col_counts) if c == max_count]

    return {
        'n_cols': n_cols,
        'nnz': nnz,
        'empty_cols': empty_cols,
        'empty_col_indices': empty_col_indices,
        'max_count': max_count,
        'max_cols': max_cols,
        'min_count': min_count,
        'avg_count': avg_count,
        'col_counts': col_counts
    }

def print_analysis(result, show_distribution=True):
    """분석 결과를 출력합니다."""
    print("\n=== Column Pointer Analysis ===")
    print("Total columns: {}".format(result['n_cols']))
    print("Total non-zeros (nnz): {}".format(result['nnz']))
    print()
    print("Empty columns: {}".format(result['empty_cols']))
    print("Non-empty columns: {}".format(result['n_cols'] - result['empty_cols']))
    print()
    print("Max elements per column: {}".format(result['max_count']))
    print("  Columns with max: {} (first 10: {})".format(
        len(result['max_cols']),
        result['max_cols'][:10]
    ))
    print("Min elements per column: {}".format(result['min_count']))
    print("Avg elements per column: {:.2f}".format(result['avg_count']))

    if result['empty_cols'] > 0 and result['empty_cols'] <= 30:
        print()
        print("Empty column indices: {}".format(result['empty_col_indices']))

    if show_distribution:
        print()
        print("Distribution of elements per column:")
        from collections import Counter
        dist = Counter(result['col_counts'])
        sorted_keys = sorted(dist.keys())
        for k in sorted_keys[:25]:
            print("  {:3d} elements: {:4d} columns".format(k, dist[k]))
        if len(sorted_keys) > 25:
            print("  ... ({} more unique counts)".format(len(sorted_keys) - 25))

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_col_ptr.py <col_ptr_file.hex>")
        print("Example: python analyze_col_ptr.py Ap.hex")
        print("         python analyze_col_ptr.py Lp.txt")
        return

    filepath = sys.argv[1]
    print("Loading: {}".format(filepath))

    col_ptr = load_col_ptr(filepath)
    print("Loaded {} entries".format(len(col_ptr)))

    result = analyze_col_ptr(col_ptr)
    print_analysis(result)

    return result

if __name__ == "__main__":
    main()
