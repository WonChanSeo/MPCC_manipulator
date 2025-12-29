# -*- coding: utf-8 -*-
"""
CSC (Compressed Sparse Column) to CSR (Compressed Sparse Row) 변환

CSC format:
- col_ptr (Lp): 각 열의 시작 인덱스
- row_indices (Li): 각 non-zero의 행 인덱스
- values (Lx): non-zero 값들

CSR format:
- row_ptr: 각 행의 시작 인덱스
- col_indices: 각 non-zero의 열 인덱스
- values: non-zero 값들
"""
import sys

def load_array(filepath):
    """파일에서 숫자 배열을 읽어옵니다."""
    values = []
    is_hex = None

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return values

    # 첫 몇 줄을 보고 hex인지 decimal인지 판단
    # hex 파일은 보통 8자리 또는 16자리 고정 길이
    sample = lines[0]
    if len(sample) == 8 or len(sample) == 16:
        # 고정 길이이고 모두 hex 문자면 hex로 간주
        if all(c in '0123456789abcdefABCDEF' for c in sample):
            is_hex = True

    # .hex 확장자면 hex로 간주
    if filepath.endswith('.hex'):
        is_hex = True

    for line in lines:
        if is_hex:
            values.append(int(line, 16))
        else:
            try:
                # 소수점이 있으면 float
                if '.' in line:
                    values.append(float(line))
                else:
                    values.append(int(line))
            except:
                values.append(0)

    return values

def load_li_by_col(filepath):
    """Li_by_col.txt 형식 파일을 읽어옵니다."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row_indices = [int(x) for x in line.split(',')]
                data.append(row_indices)
            else:
                data.append([])
    return data

def csc_to_csr(col_ptr, row_indices, values=None, n_rows=None):
    """
    CSC 형식을 CSR 형식으로 변환합니다.

    Parameters:
    - col_ptr: 열 포인터 배열 (길이: n_cols + 1)
    - row_indices: 행 인덱스 배열
    - values: 값 배열 (optional)
    - n_rows: 행 개수 (None이면 row_indices의 max + 1)

    Returns:
    - row_ptr: 행 포인터 배열
    - col_indices: 열 인덱스 배열
    - csr_values: 값 배열 (values가 주어진 경우)
    """
    # 정수로 변환
    col_ptr = [int(x) for x in col_ptr]
    row_indices = [int(x) for x in row_indices]

    n_cols = len(col_ptr) - 1
    nnz = len(row_indices)

    print("Debug: n_cols={}, nnz={}, col_ptr[-1]={}".format(n_cols, nnz, col_ptr[-1]))

    if n_rows is None:
        n_rows = max(row_indices) + 1 if row_indices else n_cols

    print("Debug: n_rows={}, max(row_indices)={}".format(n_rows, max(row_indices) if row_indices else 0))

    # 1. 각 행에 몇 개의 non-zero가 있는지 카운트
    row_counts = [0] * n_rows
    for row in row_indices:
        if row < n_rows:
            row_counts[int(row)] += 1

    # 2. row_ptr 생성 (cumulative sum)
    row_ptr = [0] * (n_rows + 1)
    for i in range(n_rows):
        row_ptr[i + 1] = row_ptr[i] + row_counts[i]

    # 3. col_indices와 values 배열 생성
    col_indices = [0] * nnz
    csr_values = [0.0] * nnz if values else None

    # 현재 각 행에서 다음에 채울 위치
    current_pos = row_ptr[:]

    # CSC를 순회하면서 CSR 채우기
    for col in range(n_cols):
        start = int(col_ptr[col])
        end = int(col_ptr[col + 1])

        for idx in range(start, end):
            if idx >= len(row_indices):
                print("Warning: idx {} out of range (len={})".format(idx, len(row_indices)))
                continue
            row = int(row_indices[idx])
            if row < n_rows:
                dest = current_pos[row]
                col_indices[dest] = col
                if values and idx < len(values):
                    csr_values[dest] = values[idx]
                current_pos[row] += 1

    return row_ptr, col_indices, csr_values

def save_csr_files(row_ptr, col_indices, values, prefix):
    """CSR 데이터를 파일로 저장합니다."""
    # row_ptr 저장
    with open(prefix + '_row_ptr.txt', 'w') as f:
        for val in row_ptr:
            f.write('{}\n'.format(int(val)))

    # col_indices 저장 (flat)
    with open(prefix + '_col_indices.txt', 'w') as f:
        for val in col_indices:
            f.write('{}\n'.format(int(val)))

    # col_indices를 행별로 그룹화하여 저장
    with open(prefix + '_col_indices_by_row.txt', 'w') as f:
        for row in range(len(row_ptr) - 1):
            start = row_ptr[row]
            end = row_ptr[row + 1]
            indices = col_indices[start:end]
            line = ','.join(str(int(idx)) for idx in indices)
            f.write(line + '\n')

    # values 저장 (있는 경우)
    if values:
        with open(prefix + '_values.txt', 'w') as f:
            for val in values:
                f.write('{}\n'.format(val))

        with open(prefix + '_values_by_row.txt', 'w') as f:
            for row in range(len(row_ptr) - 1):
                start = row_ptr[row]
                end = row_ptr[row + 1]
                vals = values[start:end]
                line = ','.join(str(v) for v in vals)
                f.write(line + '\n')

    print("Saved: {}_row_ptr.txt".format(prefix))
    print("Saved: {}_col_indices.txt".format(prefix))
    print("Saved: {}_col_indices_by_row.txt".format(prefix))
    if values:
        print("Saved: {}_values.txt".format(prefix))
        print("Saved: {}_values_by_row.txt".format(prefix))

def print_csr_info(row_ptr, col_indices, values):
    """CSR 정보를 출력합니다."""
    n_rows = len(row_ptr) - 1
    nnz = len(col_indices)

    print("\n=== CSR Format Info ===")
    print("Number of rows: {}".format(n_rows))
    print("Number of non-zeros (nnz): {}".format(nnz))
    print("row_ptr length: {}".format(len(row_ptr)))
    print("col_indices length: {}".format(len(col_indices)))

    print("\nFirst 10 row_ptr: {}".format(row_ptr[:10]))
    print("First 10 col_indices: {}".format(col_indices[:10]))
    if values:
        print("First 10 values: {}".format(values[:10]))

    # 각 행의 nnz 분포
    row_nnz = [row_ptr[i+1] - row_ptr[i] for i in range(min(10, n_rows))]
    print("\nFirst 10 rows' nnz counts: {}".format(row_nnz))

    max_nnz = max(row_ptr[i+1] - row_ptr[i] for i in range(n_rows))
    print("Maximum nnz per row: {}".format(max_nnz))

def main():
    base_dir = './qdldl_permuted_output/'

    if len(sys.argv) >= 2:
        base_dir = sys.argv[1]
        if not base_dir.endswith('/'):
            base_dir += '/'

    # CSC 파일들 로드
    print("Loading CSC data from: {}".format(base_dir))

    try:
        col_ptr = load_array(base_dir + 'Lp.txt')
        print("Loaded Lp.txt: {} entries".format(len(col_ptr)))
    except:
        print("Error: Cannot load Lp.txt")
        return

    try:
        row_indices = load_array(base_dir + 'Li.txt')
        print("Loaded Li.txt: {} entries".format(len(row_indices)))
    except:
        print("Error: Cannot load Li.txt")
        return

    # Lx (values) 로드 시도
    values = None
    try:
        values = load_array(base_dir + 'Lx.txt')
        print("Loaded Lx.txt: {} entries".format(len(values)))
    except:
        print("Lx.txt not found, proceeding without values")

    # 행 개수 결정 (정방행렬 가정: n_rows = n_cols)
    n_cols = len(col_ptr) - 1
    n_rows = n_cols
    print("\nMatrix size: {} x {}".format(n_rows, n_cols))

    # CSC -> CSR 변환
    print("\nConverting CSC to CSR...")
    row_ptr, col_indices, csr_values = csc_to_csr(
        col_ptr, row_indices, values, n_rows
    )

    # 정보 출력
    print_csr_info(row_ptr, col_indices, csr_values)

    # 파일 저장
    output_prefix = base_dir + 'L_csr'
    save_csr_files(row_ptr, col_indices, csr_values, output_prefix)

    print("\nConversion complete!")
    return row_ptr, col_indices, csr_values

if __name__ == "__main__":
    main()
