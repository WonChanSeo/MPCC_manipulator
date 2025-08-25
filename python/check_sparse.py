#!/usr/bin/env python3
import re
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

def plot_sparsity_from_file(filename: str):
    # 1) 파일 전체 읽기
    with open(filename, 'r') as f:
        content = f.read()

    # 2) Nonzero entries / Outer pointers 텍스트 분리
    nz_text    = content.split('Nonzero entries:')[1].split('Outer pointers:')[0]
    outer_text = content.split('Outer pointers:')[1].split('$')[0]

    # 3) (value, row) 페어 파싱
    pairs  = re.findall(r'\(\s*([\-0-9.eE+]+)\s*,\s*([0-9]+)\s*\)', nz_text)
    values = [float(v) for v, _ in pairs]
    rows   = [int(r)   for _, r in pairs]

    # 4) outer pointer 파싱
    outer = [int(x) for x in re.findall(r'\d+', outer_text)]

    # 5) 기대 non-zero 개수 기준으로 trimming
    nnz_expected = outer[-1]
    if len(values) > nnz_expected:
        print(f"[WARNING] trimming {len(values) - nnz_expected} extra entries")
        values = values[:nnz_expected]
        rows   = rows[:nnz_expected]

    # 6) 행/열 크기 계산
    n_cols = len(outer) - 1
    n_rows = max(rows) + 1

    # 7) cols 인덱스 재생성
    cols = []
    for col in range(n_cols):
        cnt = outer[col+1] - outer[col]
        cols.extend([col] * cnt)

    # 8) 길이 일치 확인
    nnz = len(values)
    if not (nnz == len(rows) == len(cols) == nnz_expected):
        raise ValueError(
            f"still mismatch: values={len(values)}, rows={len(rows)}, cols={len(cols)}, expected={nnz_expected}"
        )

    # 9) CSC 행렬 생성 및 시각화
    mat = csc_matrix((values, (rows, cols)), shape=(n_rows, n_cols))
    plt.figure(figsize=(6,6))
    plt.spy(mat, precision=0, markersize=1)
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.title('Sparsity pattern (non-zero entries)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_sparsity_from_file('./P_sp.txt')
