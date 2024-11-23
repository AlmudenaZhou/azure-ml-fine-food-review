import os
import argparse

import pandas as pd

from text_processing_step import TextPreprocessingStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--processed_filename", type=str, help="name of the posprocessed file", default="data.csv")
    parser.add_argument("--processed_data", type=str, help="path to posprocessed data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.input_data)

    data = pd.read_csv(args.input_data, header=1)

    processed_data = TextPreprocessingStep().main(data)

    processed_data.to_csv(os.path.join(args.processed_data, args.processed_filename), index=False)


if __name__ == "__main__":
    main()
