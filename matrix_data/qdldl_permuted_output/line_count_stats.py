import os
from collections import Counter

def analyze_file(filepath):
    """각 라인의 요소 개수를 분석"""
    counts = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line == '':
                count = 0
            else:
                count = len(line.split(','))
            counts.append(count)

    return counts

def print_stats(name, counts):
    """Print statistics"""
    print(f"\n{'='*60}")
    print(f"File: {name}")
    print(f"{'='*60}")

    total_lines = len(counts)
    total_elements = sum(counts)
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    avg_count = total_elements / total_lines if total_lines > 0 else 0

    # Count distribution
    count_dist = Counter(counts)

    print(f"\n[Basic Stats]")
    print(f"  Total lines: {total_lines}")
    print(f"  Total elements: {total_elements}")
    print(f"  Min count: {min_count}")
    print(f"  Max count: {max_count}")
    print(f"  Avg count: {avg_count:.2f}")

    # Empty lines
    empty_lines = count_dist.get(0, 0)
    print(f"  Empty lines: {empty_lines} ({100*empty_lines/total_lines:.1f}%)")

    print(f"\n[Distribution]")
    print(f"  {'Count':<8} {'Lines':<10} {'Pct':<10} {'Cumul':<10}")
    print(f"  {'-'*38}")

    cumulative = 0
    for count in sorted(count_dist.keys()):
        freq = count_dist[count]
        cumulative += freq
        pct = 100 * freq / total_lines
        cum_pct = 100 * cumulative / total_lines
        print(f"  {count:<8} {freq:<10} {pct:>6.1f}%    {cum_pct:>6.1f}%")

    # 히스토그램 (간단한 텍스트 버전)
    print(f"\n[Histogram]")
    max_freq = max(count_dist.values())
    bar_width = 40
    for count in sorted(count_dist.keys()):
        freq = count_dist[count]
        bar_len = int(bar_width * freq / max_freq)
        bar = '#' * bar_len
        print(f"  {count:>3}: {bar} ({freq})")

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))

    files = [
        'Li_by_col.txt',
        'L_csr_col_indices_by_row.txt'
    ]

    for filename in files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            counts = analyze_file(filepath)
            print_stats(filename, counts)
        else:
            print(f"\nFile not found: {filepath}")

if __name__ == '__main__':
    main()
