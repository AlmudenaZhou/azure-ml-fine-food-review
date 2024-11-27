import argparse

from register_model_step import RegisterModelStep




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    model_path = args.model_path + "/model.pkl"

    RegisterModelStep().main(model_path)




if __name__ == "__main__":
    main()
