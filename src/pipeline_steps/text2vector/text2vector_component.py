import os
import argparse

import pandas as pd

from text2vector_step import Text2VectorStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_x", type=str, help="path to features data")
    parser.add_argument("--input_y", type=str, help="path to label data")
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--output_folder_path", type=str, help="path to the output folder")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.input_x, " & ", args.input_y)

    X_train = pd.Series(pd.read_csv(args.input_x).iloc[:, 0])
    y_train = pd.Series(pd.read_csv(args.input_y).iloc[:, 0])

    X_train, _ = Text2VectorStep(model_path=args.model_path).main_train(X_train, y_train)

    X_train.to_csv(os.path.join(args.output_folder_path, "text2vect_X_train.csv"), index=False)


if __name__ == "__main__":
    main()
