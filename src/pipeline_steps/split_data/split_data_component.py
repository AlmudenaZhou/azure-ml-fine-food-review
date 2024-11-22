import os
import argparse

import pandas as pd

from split_data_step import SplitDataStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--split_data", type=str, help="path to split data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.input_data)

    data = pd.read_csv(args.input_data)

    X_train, X_test, y_train, y_test = SplitDataStep().main(data)

    X_train.to_csv(os.path.join(args.split_data, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(args.split_data, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(args.split_data, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(args.split_data, "y_test.csv"), index=False)


if __name__ == "__main__":
    main()
