from utils.io.io import read_file, write_file

# print tsv file in a human-readable format
def print_tsv(tsv_file, max_lines=10):
    lines = read_file(tsv_file)
    for line in lines[:max_lines]:
        items = line.rstrip().split("\t")
        for item in items:
            print(item)
    print(f"Total lines: {len(lines)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_file", type=str, default="data/cfq/origin_t5/mcd1/train.tsv")
    args = parser.parse_args()
    print_tsv(args.tsv_file)

if __name__ == "__main__":
    main()