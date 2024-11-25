import os
import argparse

import pandas as pd

from handle_imbalance_step import HandleImbalanceStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_x", type=str, help="path to features data")
    parser.add_argument("--input_y", type=str, help="path to label data")
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--output_folder_path", type=str, help="path to the output folder")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.input_x, " & ", args.input_y)

    X_train = pd.read_csv(args.input_x)
    y_train = pd.Series(pd.read_csv(args.input_y).iloc[:, 0])

    X_train, y_train = HandleImbalanceStep(model_path=args.model_path).main(X_train, y_train)

    X_train.to_csv(os.path.join(args.output_folder_path, "imb_X_train.csv"), index=False)
    y_train.to_csv(os.path.join(args.output_folder_path, "imb_y_train.csv"), index=False)


if __name__ == "__main__":
    main()
