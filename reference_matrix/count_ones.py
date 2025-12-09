import csv
import os

def count_ones_in_csv(filepath):
    count = 0
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for val in row:
                if val.strip() == '1':
                    count += 1
    return count

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "KKT_mask.csv")

    ones_count = count_ones_in_csv(filepath)
    print(f"Number of 1s in KKT_mask.csv: {ones_count}")
