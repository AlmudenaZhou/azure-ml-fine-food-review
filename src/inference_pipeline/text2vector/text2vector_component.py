import os
import argparse

import pandas as pd

from text2vector_step import Text2VectorStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model loaded")
    parser.add_argument("--input_data_folder", type=str, help="input data folder")
    parser.add_argument("--input_data_filename", type=str, help="input data file name")
    parser.add_argument("--output_data_folder", type=str, help="path to the output data")
    parser.add_argument("--output_data_filename", type=str, help="file name of the output data")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data_x = pd.Series(pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename)))

    vector_data = Text2VectorStep().main(model=args.model, data_x=data_x)

    vector_data.to_csv(os.path.join(args.output_data_folder, args.output_data_filename), index=False)


if __name__ == "__main__":
    main()
