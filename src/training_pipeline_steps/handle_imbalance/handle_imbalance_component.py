import os
import argparse

import pandas as pd

from handle_imbalance_step import HandleImbalanceStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to data")
    parser.add_argument("--input_x_filename", type=str, help="filename to features data")
    parser.add_argument("--input_y_filename", type=str, help="filename to label data")

    parser.add_argument("--imb_x_data_filename", type=str, help="path to x train modified",
                        default="imb_X_train.csv", required=False)
    parser.add_argument("--imb_y_data_filename", type=str, help="path to y train modified",
                        default="imb_y_train.csv", required=False)

    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--output_folder_path", type=str, help="path to the output folder")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    X_train = pd.read_csv(os.path.join(args.input_data_folder, args.input_x_filename))
    y_train = pd.Series(pd.read_csv(os.path.join(args.input_data_folder, args.input_y_filename)).iloc[:, 0])

    X_train, y_train = HandleImbalanceStep(model_path=args.model_path).main(X_train, y_train)

    X_train.to_csv(os.path.join(args.output_folder_path, args.imb_x_data_filename), index=False)
    y_train.to_csv(os.path.join(args.output_folder_path, args.imb_y_data_filename), index=False)


if __name__ == "__main__":
    main()
