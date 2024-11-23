import os
import argparse

import pandas as pd

from src.pipeline_steps.training_data_cleaning.training_data_cleaning_step import TrainingDataCleaningStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--clean_filename", type=str, help="name of the cleaned data file", default="data.csv")
    parser.add_argument("--clean_data", type=str, help="path to cleaned data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)

    data = pd.read_csv(args.data, header=1)

    cleaned_data = TrainingDataCleaningStep().main(data)

    cleaned_data.to_csv(os.path.join(args.clean_data, args.clean_filename), index=False)


if __name__ == "__main__":
    main()
