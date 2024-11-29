import os
import argparse

import pandas as pd

from training_data_cleaning_step import TrainingDataCleaningStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="name of the input file data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")
    parser.add_argument("--output_data_folder", type=str, help="path to cleaned data")
    parser.add_argument("--output_data_filename", type=str, help="path to cleaned data",
                        required=False, default="cleaned_data.csv")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data = pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename))

    cleaned_data = TrainingDataCleaningStep().main(data)

    cleaned_data.to_csv(os.path.join(args.output_data_folder, args.output_data_filename), index=False)


if __name__ == "__main__":
    main()
