
from src.tools.azure_ml_utils import (create_azure_component, generic_running_component_inputs_outputs,
                                      generic_creation_component_inputs_outputs, run_azure_component)


def create_training_data_cleaning_component():
    component_name = "training_data_cleaning"
    display_name = "Data cleaning for training"
    description = "Reads a .csv file and cleans duplicates and converts the labels"
    code_folder = "./src/pipeline_components/training_data_cleaning"
    code_filename = "training_data_cleaning_component.py"
    inputs, outputs, code_command = generic_creation_component_inputs_outputs(code_filename)
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_training_data_cleaning_component(wait_for_completion=False):
    input_data_filename = "reviews_short.csv"
    output_data_filename = "cleaned_data.csv"
    inputs, outputs = generic_running_component_inputs_outputs(input_data_filename, output_data_filename)
    run_azure_component(component_name="training_data_cleaning", inputs=inputs, outputs=outputs,
                        wait_for_completion=wait_for_completion)


def main():
    create_training_data_cleaning_component()
    run_training_data_cleaning_component(wait_for_completion=False)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
