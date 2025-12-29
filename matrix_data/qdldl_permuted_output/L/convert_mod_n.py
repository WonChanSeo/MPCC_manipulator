import os
import sys

def process_file(input_path, output_path, n):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        line = line.rstrip('\n')
        if line.strip():
            # Split by comma and convert each number to mod n
            numbers = [int(x.strip()) for x in line.split(',') if x.strip()]
            mod_numbers = [f'({num % n},{num // n})' for num in numbers]
            result.append(','.join(mod_numbers))
        else:
            result.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
        if result:  # Add trailing newline if there's content
            f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <n>')
        print(f'Example: python {sys.argv[0]} 6')
        sys.exit(1)

    n = int(sys.argv[1])
    base_path = os.path.dirname(os.path.abspath(__file__))

    process_file(f'{base_path}/upper_csr.txt', f'{base_path}/upper_csr_mod{n}.txt', n)
    process_file(f'{base_path}/lower_csr.txt', f'{base_path}/lower_csr_mod{n}.txt', n)
    print(f'Done! Created upper_csr_mod{n}.txt and lower_csr_mod{n}.txt')
