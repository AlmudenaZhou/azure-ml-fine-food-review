import os
import argparse

import pandas as pd

from model_prediction_step import ModelPredictionStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model loaded")
    parser.add_argument("--input_data_folder", type=str, help="input data folder")
    parser.add_argument("--input_data_filename", type=str, help="input data filname")

    parser.add_argument("--output_data_filename", type=str, help="filename for the output file")
    
    parser.add_argument("--output_data_folder", type=str, help="path to the output data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data_x = pd.read_csv(os.path.join(args.input_data_filename, ))

    preds = ModelPredictionStep().main(model=args.model, data_x=data_x)

    preds.to_csv(os.path.join(args.output_data_folder, args.output_data_filename), index=False)


if __name__ == "__main__":
    main()
