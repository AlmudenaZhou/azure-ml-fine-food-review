import os
from src.tools.azure_ml_utils import (create_azure_component, generic_creation_component_inputs_outputs,
                                      generic_running_component_inputs_outputs, run_azure_component)


def create_train_predict_model_component():

    component_name = "train_predict_model"
    display_name = "Simple model training"
    description = "Tests 3 types of models and chooses the best one. Not scalable to big amount of data."
    code_folder = "./src/pipeline_components/train_predict_model"
    code_filename = "train_predict_model_component.py"

    inputs, outputs, code_command = generic_creation_component_inputs_outputs(code_filename, is_model=True)
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_train_predict_model_component(wait_for_completion=False):
    input_data_filename = "handled_imb_data.csv"
    output_data_filename = "predictions.csv"
    model_filename = "predictor.pkl"
    inputs, outputs = generic_running_component_inputs_outputs(input_data_filename, output_data_filename,
                                                               model_filename, is_model=True)
    environment_variables = {"SCORING": os.environ["SCORING"], "TARGET": "Label"}
    run_azure_component("train_predict_model", inputs, outputs, wait_for_completion, environment_variables)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_train_predict_model_component()
