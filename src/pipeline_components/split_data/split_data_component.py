import os
import argparse

import pandas as pd

from split_data_step import SplitDataStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to input data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")

    parser.add_argument("--x_train_filename", type=str, help="path to x train data", default="X_train.csv")
    parser.add_argument("--x_test_filename", type=str, help="path to x test data", default="X_test.csv")
    parser.add_argument("--y_train_filename", type=str, help="path to y train data", default="y_train.csv")
    parser.add_argument("--y_test_filename", type=str, help="path to y test data", default="y_test.csv")

    parser.add_argument("--split_data", type=str, help="path to split data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data = pd.read_csv(args.input_data_folder + "/" + args.input_data_filename)

    X_train, X_test, y_train, y_test = SplitDataStep().main(data)

    X_train.to_csv(args.split_data + "/" + args.x_train_filename, index=False)
    X_test.to_csv(args.split_data + "/" + args.x_test_filename, index=False)
    y_train.to_csv(args.split_data + "/" + args.y_train_filename, index=False)
    y_test.to_csv(args.split_data + "/" + args.y_test_filename, index=False)


if __name__ == "__main__":
    main()
