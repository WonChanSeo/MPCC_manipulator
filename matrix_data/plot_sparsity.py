# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sys

def load_li_by_col(filepath):
    """Li_by_col.txt 파일을 읽어 (col, row) 쌍을 반환합니다."""
    col_indices = []
    row_indices = []

    with open(filepath, 'r') as f:
        for col, line in enumerate(f):
            line = line.strip()
            if line:
                rows = [int(x) for x in line.split(',')]
                for row in rows:
                    col_indices.append(col)
                    row_indices.append(row)

    return col_indices, row_indices

def load_csc_format(lp_path, li_path):
    """Lp.txt와 Li.txt에서 CSC 형식을 읽어 (col, row) 쌍을 반환합니다."""
    # Load Lp (column pointers)
    col_ptr = []
    with open(lp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                col_ptr.append(int(line))

    # Load Li (row indices)
    row_indices_flat = []
    with open(li_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row_indices_flat.append(int(line))

    # Convert to (col, row) pairs
    col_indices = []
    row_indices = []
    num_cols = len(col_ptr) - 1

    for col in range(num_cols):
        start = col_ptr[col]
        end = col_ptr[col + 1]
        for idx in range(start, end):
            col_indices.append(col)
            row_indices.append(row_indices_flat[idx])

    return col_indices, row_indices

def plot_sparsity(col_indices, row_indices, output_path, title="Sparsity pattern (non-zero entries)"):
    """Sparsity pattern을 시각화합니다."""
    plt.figure(figsize=(10, 8))
    plt.scatter(col_indices, row_indices, s=1, marker='.')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.title(title)
    plt.gca().invert_yaxis()  # Row 0을 상단에 표시
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved plot to: {}".format(output_path))

def main():
    # 기본 경로 설정
    base_dir = './qdldl_permuted_output/'

    if len(sys.argv) >= 2:
        # Li_by_col.txt 형식
        li_by_col_path = sys.argv[1]
        print("Loading from: {}".format(li_by_col_path))
        col_indices, row_indices = load_li_by_col(li_by_col_path)
        output_path = li_by_col_path.replace('.txt', '_sparsity.png')
    else:
        # CSC 형식 (Lp.txt, Li.txt)
        lp_path = base_dir + 'Lp.txt'
        li_path = base_dir + 'Li.txt'

        # Li_by_col.txt가 있으면 사용
        try:
            li_by_col_path = base_dir + 'Li_by_col.txt'
            print("Loading from: {}".format(li_by_col_path))
            col_indices, row_indices = load_li_by_col(li_by_col_path)
            output_path = base_dir + 'L_sparsity.png'
        except:
            print("Loading from Lp.txt and Li.txt")
            col_indices, row_indices = load_csc_format(lp_path, li_path)
            output_path = base_dir + 'L_sparsity.png'

    print("Number of non-zeros: {}".format(len(col_indices)))
    print("Column range: {} - {}".format(min(col_indices), max(col_indices)))
    print("Row range: {} - {}".format(min(row_indices), max(row_indices)))

    plot_sparsity(col_indices, row_indices, output_path)

if __name__ == "__main__":
    main()
