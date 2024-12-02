import os
import argparse

import pandas as pd

from text_processing_step import TextPreprocessingStep


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, help="path to data")
    parser.add_argument("--input_data_filename", type=str, help="name of the input file data")
    parser.add_argument("--output_data_folder", type=str, help="path to posprocessed data")
    parser.add_argument("--output_data_filename", type=str, help="path to posprocessed data",
                        required=False, default="processed_data.csv")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data = pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename))

    text_colname = os.getenv("TEXT_COLNAME", "Text")
    processed_data = TextPreprocessingStep(text_colname).main(data)

    processed_data.to_csv(os.path.join(args.output_data_folder, args.output_data_filename), index=False)


if __name__ == "__main__":
    main()
