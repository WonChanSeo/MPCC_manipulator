import re

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        line = line.rstrip('\n')
        if line.strip():
            # Split by comma and convert each number to mod 6
            numbers = [int(x.strip()) for x in line.split(',') if x.strip()]
            mod_numbers = [str(n % 6) for n in numbers]
            result.append(','.join(mod_numbers))
        else:
            result.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
        if result:  # Add trailing newline if there's content
            f.write('\n')

if __name__ == '__main__':
    import os

    base_path = os.path.dirname(os.path.abspath(__file__))
    process_file(f'{base_path}/upper_csr.txt', f'{base_path}/upper_csr_mod6.txt')
    process_file(f'{base_path}/lower_csr.txt', f'{base_path}/lower_csr_mod6.txt')
    print('Done!')
