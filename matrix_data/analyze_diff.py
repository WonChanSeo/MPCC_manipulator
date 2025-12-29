# -*- coding: utf-8 -*-
import sys

def load_values(filepath):
    """파일에서 숫자들을 읽어옵니다."""
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))
    return values

def analyze_differences(values):
    """인접한 값들의 차이를 분석합니다."""
    if len(values) < 2:
        return None

    differences = []
    for i in range(len(values) - 1):
        diff = values[i + 1] - values[i]
        differences.append({
            'index': i,
            'value1': values[i],
            'value2': values[i + 1],
            'diff': diff
        })

    # 최대 차이 찾기
    max_diff_entry = max(differences, key=lambda x: x['diff'])

    # 차이가 큰 순으로 정렬
    sorted_diffs = sorted(differences, key=lambda x: x['diff'], reverse=True)

    return {
        'differences': differences,
        'max_diff': max_diff_entry,
        'top_diffs': sorted_diffs[:20]  # 상위 20개
    }

def save_results(result, output_path):
    """결과를 파일로 저장합니다."""
    with open(output_path, 'w') as f:
        f.write("=== Maximum Difference ===\n")
        max_d = result['max_diff']
        f.write("Index: {} (line {} -> line {})\n".format(max_d['index'], max_d['index'] + 1, max_d['index'] + 2))
        f.write("Values: {} -> {}\n".format(max_d['value1'], max_d['value2']))
        f.write("Difference: {}\n\n".format(max_d['diff']))

        f.write("=== Top 20 Largest Differences ===\n")
        f.write("{:>8} {:>12} {:>12} {:>10}\n".format("Index", "Value1", "Value2", "Diff"))
        f.write("-" * 45 + "\n")
        for d in result['top_diffs']:
            f.write("{:>8} {:>12} {:>12} {:>10}\n".format(
                d['index'], d['value1'], d['value2'], d['diff']))

def main():
    if len(sys.argv) < 2:
        filepath = './qdldl_permuted_output/Lp.txt'
    else:
        filepath = sys.argv[1]

    print("Loading values from: {}".format(filepath))
    values = load_values(filepath)
    print("Total values: {}".format(len(values)))

    print("Analyzing differences...")
    result = analyze_differences(values)

    if result is None:
        print("Not enough values to analyze.")
        return

    # 결과 출력
    max_d = result['max_diff']
    print("\n=== Maximum Difference ===")
    print("Index: {} (line {} -> line {})".format(max_d['index'], max_d['index'] + 1, max_d['index'] + 2))
    print("Values: {} -> {}".format(max_d['value1'], max_d['value2']))
    print("Difference: {}".format(max_d['diff']))

    print("\n=== Top 10 Largest Differences ===")
    print("{:>8} {:>12} {:>12} {:>10}".format("Index", "Value1", "Value2", "Diff"))
    print("-" * 45)
    for d in result['top_diffs'][:10]:
        print("{:>8} {:>12} {:>12} {:>10}".format(
            d['index'], d['value1'], d['value2'], d['diff']))

    # 결과 저장
    output_path = filepath.replace('.txt', '_diff_analysis.txt')
    save_results(result, output_path)
    print("\nResults saved to: {}".format(output_path))

    return result

if __name__ == "__main__":
    result = main()
