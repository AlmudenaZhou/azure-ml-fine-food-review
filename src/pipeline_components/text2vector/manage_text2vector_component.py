import os

from src.tools.azure_ml_utils import (create_azure_component, generic_creation_component_inputs_outputs,
                                      generic_running_component_inputs_outputs, run_azure_component)


def create_text2vector_component():
    component_name = "text2vector"
    display_name = "Text to vector dataframe"
    description = ("Gets a csv with X_train and y_train and converts the X_train to vector"
                   "and saves the model used.")
    code_folder = "./src/pipeline_components/text2vector"
    code_filename = "text2vector_component.py"
    inputs, outputs, code_command = generic_creation_component_inputs_outputs(code_filename, is_model=True)
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_text2vector_component(wait_for_completion=False):
    input_data_filename = "train_data.csv"
    output_data_filename = "text2vector_data.csv"
    model_filename = "text2vector.pkl"
    inputs, outputs = generic_running_component_inputs_outputs(input_data_filename, output_data_filename,
                                                               model_filename, is_model=True)
    environment_variables = {"SCORING": os.environ["SCORING"]}
    run_azure_component("text2vector", inputs, outputs, wait_for_completion, environment_variables)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_text2vector_component()
