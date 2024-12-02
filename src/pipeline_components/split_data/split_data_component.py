import os
import argparse

import pandas as pd

from split_data_step import SplitDataStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to input data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")
    parser.add_argument("--train_output_filename", type=str, help="path to x train data", default="train_data.csv")
    parser.add_argument("--test_output_filename", type=str, help="path to x test data", default="test_data.csv")

    parser.add_argument("--output_data_folder", type=str, help="path to split data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data = pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename))

    train_data, test_data = SplitDataStep(target="Label").main(data)

    train_data.to_csv(os.path.join(args.output_data_folder, args.train_output_filename), index=False)
    test_data.to_csv(os.path.join(args.output_data_folder, args.test_output_filename), index=False)


if __name__ == "__main__":
    main()
