import os
import argparse

import pandas as pd

from train_predict_model_step import TrainPredictStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")
    parser.add_argument("--output_data_folder", type=str, help="path to predicted data")
    parser.add_argument("--output_data_filename", type=str, help="filename to predicted data",
                        required=False, default="predictions.csv")
    parser.add_argument("--model_filename", type=str, help="model file name")
    parser.add_argument("--model_input_path", type=str, default="", required=False,
                        help="path to load the model if exists")
    parser.add_argument("--model_output_folder", type=str, help="folder to save the model")
    parser.add_argument("--is_training", type=str, help="If the component is for training, value `True`")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    train_data = pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename))

    target = os.getenv("TARGET", "Label")
    model_path = args.model_input_path
    if not args.model_input_path:
        model_path = os.path.join(args.model_output_folders, args.model_filename)
    is_training = args.is_training == "True"
    preds = TrainPredictStep(target, model_path=model_path).main(train_data, is_training)

    preds.to_csv(os.path.join(args.output_data_folder, args.output_data_filename))


if __name__ == "__main__":
    main()
