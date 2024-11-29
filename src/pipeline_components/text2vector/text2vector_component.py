import os
import argparse

import pandas as pd

from text2vector_step import Text2VectorStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to data")
    parser.add_argument("--input_x_filename", type=str, help="filename to features data")
    parser.add_argument("--input_y_filename", type=str, help="filename to label data")
    parser.add_argument("--text2vec_data_filename", type=str, help="path to x train modified",
                        default="text2vect_X_train.csv", required=False)
    parser.add_argument("--text2vec_model_filename", type=str, help="path to save the trained model",
                        default="text2vect.pkl", required=False)
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--output_folder_path", type=str, help="path to the output folder")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    X_train = pd.Series(pd.read_csv(args.input_data_folder + "/" + args.input_x_filename).iloc[:, 0])
    y_train = pd.Series(pd.read_csv(args.input_data_folder + "/" + args.input_y_filename).iloc[:, 0])

    model_path = os.path.join(args.model_path, args.text2vec_model_filename)
    X_train, _ = Text2VectorStep(model_path=model_path).main_train(X_train, y_train)

    X_train.to_csv(os.path.join(args.output_folder_path, args.text2vec_data_filename), index=False)
    y_train.to_csv(os.path.join(args.output_folder_path, args.input_y_filename), index=False)


if __name__ == "__main__":
    main()
