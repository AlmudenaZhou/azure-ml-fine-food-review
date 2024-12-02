import os
import argparse

import pandas as pd

from text2vector_step import Text2VectorStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")
    parser.add_argument("--output_data_folder", type=str, help="path to cleaned data")
    parser.add_argument("--output_data_filename", type=str, help="path to cleaned data",
                        required=False, default="text2vec_data.csv")
    parser.add_argument("--model_filename", type=str, help="model file name")
    parser.add_argument("--model_folder", type=str, help="folder to save the model")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    train_data = pd.read_csv(os.path.join(args.model_folder, args.model_filename))

    X_train = train_data.Text
    y_train = train_data.Label

    model_path = os.path.join(args.model_folder, args.model_filename)
    X_train, y_train = Text2VectorStep(model_path=model_path).main_train(X_train, y_train)

    train_data = pd.concat([X_train, y_train], axis=1)

    train_data.to_csv(os.path.join(args.output_folder_path, args.text2vec_data_filename), index=False)


if __name__ == "__main__":
    main()
