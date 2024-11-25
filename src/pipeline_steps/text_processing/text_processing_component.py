import os
import argparse

import pandas as pd

from text_processing_step import TextPreprocessingStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to input data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")
    parser.add_argument("--processed_filename", type=str, help="name of the posprocessed file", default="processed_data.csv")
    parser.add_argument("--processed_data", type=str, help="path to posprocessed data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data = pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename))

    processed_data = TextPreprocessingStep().main(data)

    processed_data.to_csv(os.path.join(args.processed_data, args.processed_filename), index=False)


if __name__ == "__main__":
    main()
