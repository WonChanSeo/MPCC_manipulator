# -*- coding: utf-8 -*-
import numpy as np
import re

def load_csv_matrix(filepath):
    """CSV 파일을 읽어 matrix를 생성합니다."""
    rows = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Find CSV part
            match = re.search(r'(\d+,[\d,]+)', line)
            if match:
                values_str = match.group(1)
                row_data = [int(v) for v in values_str.split(',')]
                rows.append(row_data)
            else:
                # Plain CSV
                values = line.split(',')
                row_data = [int(v) for v in values]
                rows.append(row_data)

    return np.array(rows, dtype=np.float64)

def matrix_to_csc(matrix):
    """
    Dense matrix를 CSC format으로 변환합니다.

    CSC format:
    - col_ptr: 각 열의 시작 인덱스 (길이: num_cols + 1)
    - row_indices: 각 non-zero 원소의 행 인덱스
    - values: non-zero 값들
    """
    num_rows, num_cols = matrix.shape

    values = []
    row_indices = []
    col_ptr = [0]

    for col in range(num_cols):
        for row in range(num_rows):
            if matrix[row, col] != 0:
                values.append(matrix[row, col])
                row_indices.append(row)
        col_ptr.append(len(values))

    return {
        'col_ptr': np.array(col_ptr, dtype=np.int32),
        'row_indices': np.array(row_indices, dtype=np.int32),
        'values': np.array(values, dtype=np.float64),
        'shape': matrix.shape
    }

def save_csc_to_files(csc_data, prefix):
    """CSC 데이터를 파일로 저장합니다."""
    np.savetxt(prefix + '_col_ptr.txt', csc_data['col_ptr'], fmt='%d')
    np.savetxt(prefix + '_row_indices.txt', csc_data['row_indices'], fmt='%d')
    np.savetxt(prefix + '_values.txt', csc_data['values'], fmt='%g')

    # row_indices를 열별로 그룹화하여 저장
    col_ptr = csc_data['col_ptr']
    row_indices = csc_data['row_indices']
    values = csc_data['values']

    with open(prefix + '_row_indices_by_col.txt', 'w') as f:
        for col in range(len(col_ptr) - 1):
            start = col_ptr[col]
            end = col_ptr[col + 1]
            indices = row_indices[start:end]
            line = ','.join(str(idx) for idx in indices)
            f.write(line + '\n')

    with open(prefix + '_values_by_col.txt', 'w') as f:
        for col in range(len(col_ptr) - 1):
            start = col_ptr[col]
            end = col_ptr[col + 1]
            vals = values[start:end]
            line = ','.join(str(int(v)) for v in vals)
            f.write(line + '\n')

    print("Saved: {}_col_ptr.txt".format(prefix))
    print("Saved: {}_row_indices.txt".format(prefix))
    print("Saved: {}_values.txt".format(prefix))
    print("Saved: {}_row_indices_by_col.txt".format(prefix))
    print("Saved: {}_values_by_col.txt".format(prefix))

def print_csc_info(csc_data):
    """CSC 정보를 출력합니다."""
    print("\n=== CSC Format Info ===")
    print("Matrix shape: {}".format(csc_data['shape']))
    print("Number of non-zeros (nnz): {}".format(len(csc_data['values'])))
    print("col_ptr length: {}".format(len(csc_data['col_ptr'])))
    print("row_indices length: {}".format(len(csc_data['row_indices'])))
    print("values length: {}".format(len(csc_data['values'])))
    print("\nFirst 10 col_ptr: {}".format(csc_data['col_ptr'][:10]))
    print("First 10 row_indices: {}".format(csc_data['row_indices'][:10]))
    print("First 10 values: {}".format(csc_data['values'][:10]))

def main():
    import sys

    if len(sys.argv) < 2:
        csv_path = './A.csv'
    else:
        csv_path = sys.argv[1]

    print("Loading matrix from: {}".format(csv_path))
    matrix = load_csv_matrix(csv_path)
    print("Matrix shape: {}".format(matrix.shape))

    print("\nConverting to CSC format...")
    csc_data = matrix_to_csc(matrix)

    print_csc_info(csc_data)

    # Save to files
    output_prefix = csv_path.replace('.csv', '_csc')
    save_csc_to_files(csc_data, output_prefix)

    return csc_data

if __name__ == "__main__":
    csc_data = main()
