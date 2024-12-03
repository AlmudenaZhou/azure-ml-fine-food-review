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
    parser.add_argument("--model_input_path", type=str, default="", required=False,
                        help="path to load the model if exists")
    parser.add_argument("--model_output_folder", type=str, help="folder to save the model")
    parser.add_argument("--is_training", type=str, help="If the component is for training, value `True`")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    text_colname = os.getenv("TEXT_COLNAME", "Text")
    target = os.getenv("TARGET", "Label")

    train_data = pd.read_csv(os.path.join(args.input_data_folder, args.input_data_filename))
    
    model_path = args.model_input_path
    if not args.model_input_path:
        model_path = os.path.join(args.model_output_folder, args.model_filename)

    is_training = args.is_training == "True"
    train_data = Text2VectorStep(text_colname=text_colname, target=target, model_path=model_path).main(train_data, is_training)

    train_data.to_csv(os.path.join(args.output_data_folder, args.output_data_filename), index=False)


if __name__ == "__main__":
    main()
