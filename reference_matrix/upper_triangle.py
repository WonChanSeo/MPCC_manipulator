# -*- coding: utf-8 -*-
import numpy as np
import re
import sys

def load_csv_matrix(filepath):
    """CSV 파일을 읽어 matrix를 생성합니다."""
    rows = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Find CSV part
            match = re.search(r'(-?\d+\.?\d*,[-\d,.\s]+)', line)
            if match:
                values_str = match.group(1)
                row_data = [float(v.strip()) for v in values_str.split(',')]
                rows.append(row_data)
            else:
                # Plain CSV
                values = line.split(',')
                row_data = [float(v.strip()) for v in values]
                rows.append(row_data)

    return np.array(rows, dtype=np.float64)

def upper_triangle(matrix, include_diagonal=True):
    """
    행렬의 upper triangle만 남기고 나머지는 0으로 만듭니다.

    include_diagonal=True: 대각선 포함 (k=0)
    include_diagonal=False: 대각선 제외 (k=1)
    """
    if include_diagonal:
        return np.triu(matrix, k=0)
    else:
        return np.triu(matrix, k=1)

def main():
    if len(sys.argv) < 2:
        csv_path = './permuted_A.csv'
    else:
        csv_path = sys.argv[1]

    # 대각선 포함 여부 (기본: 포함)
    include_diag = True
    if len(sys.argv) >= 3 and sys.argv[2] == '--no-diag':
        include_diag = False

    print("Loading matrix from: {}".format(csv_path))
    matrix = load_csv_matrix(csv_path)
    print("Matrix shape: {}".format(matrix.shape))

    print("Extracting upper triangle (include_diagonal={})...".format(include_diag))
    result = upper_triangle(matrix, include_diagonal=include_diag)

    # 출력 파일명 생성
    output_path = csv_path.replace('.csv', '_upper.csv')
    np.savetxt(output_path, result, delimiter=',', fmt='%g')
    print("Saved to: {}".format(output_path))

    # 통계 출력
    nnz_original = np.count_nonzero(matrix)
    nnz_upper = np.count_nonzero(result)
    print("\nOriginal non-zeros: {}".format(nnz_original))
    print("Upper triangle non-zeros: {}".format(nnz_upper))

    return result

if __name__ == "__main__":
    result = main()
