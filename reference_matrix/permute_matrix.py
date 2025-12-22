# -*- coding: utf-8 -*-
import numpy as np
import re

def load_permutation_matrix(filepath):
    """P_ref.txt 파일을 읽어 permutation matrix를 생성합니다."""
    cols = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cols.append(int(line))

    # permutation matrix 생성
    n = len(cols)
    max_col = max(cols)
    size = max(n, max_col + 1)
    P_ref = np.zeros((size, size), dtype=np.float64)

    for row, col in enumerate(cols):
        P_ref[row, col] = 1

    return P_ref

def load_kkt_mask(filepath):
    """KKT_mask.csv 파일을 읽어 matrix를 생성합니다."""
    rows = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Find CSV part: look for pattern like "123->0,1,0,..." or just "0,1,0,..."
            # Use regex to find the comma-separated values part
            match = re.search(r'(\d+,[\d,]+)', line)
            if match:
                values_str = match.group(1)
                row_data = [int(v) for v in values_str.split(',')]
                rows.append(row_data)

    return np.array(rows, dtype=np.float64)

def main():
    # 파일 경로 설정
    p_ref_path = r'./P_ref.txt'
    kkt_mask_path = r'./A.csv'
    
    # 행렬 로드
    print("Loading P_ref...")
    P_ref = load_permutation_matrix(p_ref_path)
    print("P_ref shape: {}".format(P_ref.shape))

    print("Loading KKT_mask...")
    KKT_mask = load_kkt_mask(kkt_mask_path)
    print("KKT_mask shape: {}".format(KKT_mask.shape))

    # 행렬 곱셈: P_ref @ KKT_mask @ P_ref^T (PKP^T)
    print("Computing P_ref @ KKT_mask @ P_ref^T...")
    P_ref_T = P_ref.T
    result = np.dot(np.dot(P_ref, KKT_mask), P_ref_T)
    print("Result shape: {}".format(result.shape))

    # 결과 저장 (선택사항)
    output_path = r'./permuted_A.csv'
    np.savetxt(output_path, result, delimiter=',', fmt='%d')
    print("Result saved to: {}".format(output_path))
    
    return result

if __name__ == "__main__":
    result = main()