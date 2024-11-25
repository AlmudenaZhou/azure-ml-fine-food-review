import os
import argparse

import pandas as pd

from model_training_step import ModelTrainingStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to data")
    parser.add_argument("--input_x_filename", type=str, help="filename to features data")
    parser.add_argument("--input_y_filename", type=str, help="filename to label data")
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--output_folder_path", type=str, help="path to the output folder")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    X_train = pd.read_csv(os.path.join(args.input_data_folder, args.input_x_filename))
    y_train = pd.Series(pd.read_csv(os.path.join(args.input_data_folder, args.input_y_filename)).iloc[:, 0])

    _ = ModelTrainingStep(model_path=args.model_path).main(X_train, y_train)


if __name__ == "__main__":
    main()
