"""Summarize a trusted SfM or ffnerf archive without dumping large tensors."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--values", action="store_true", help="Print small arrays (up to 32 elements)")
    args = parser.parse_args()
    with np.load(args.archive, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if args.values and value.size <= 32:
                print(value)


if __name__ == "__main__":
    main()
