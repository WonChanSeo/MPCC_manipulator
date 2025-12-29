import os
import sys
from collections import Counter

def analyze_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        line = line.rstrip('\n')
        if line.strip():
            numbers = [int(x.strip()) for x in line.split(',') if x.strip()]
            counter = Counter(numbers)

            # Find duplicates (count > 1)
            duplicates = {num: count for num, count in counter.items() if count > 1}

            if duplicates:
                max_count = max(duplicates.values())
                dup_nums = sorted(duplicates.keys())
                result.append(f'{max_count},{",".join(map(str, dup_nums))}')
            else:
                result.append('0,')  # No duplicates
        else:
            result.append('0,')  # Empty line

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
        if result:
            f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <input_file>')
        print(f'Example: python {sys.argv[0]} lower_csr_mod6.txt')
        sys.exit(1)

    input_file = sys.argv[1]
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, input_file)

    # Generate output filename
    name, ext = os.path.splitext(input_file)
    output_file = f'{name}_duplicates{ext}'
    output_path = os.path.join(base_path, output_file)

    analyze_file(input_path, output_path)
    print(f'Done! Created {output_file}')
    print(f'Format: max_duplicate_count,duplicate_numbers')
