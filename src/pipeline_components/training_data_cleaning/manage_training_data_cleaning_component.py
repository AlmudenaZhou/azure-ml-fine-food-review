
from tools.azure_ml_utils import create_azure_component, generic_component_inputs_outputs, generic_creation_component_inputs_outputs, run_azure_component


def create_training_data_cleaning_component(component_name, display_name, description, 
                                            code_folder, code_filename):

    inputs, outputs, code_command = generic_creation_component_inputs_outputs(code_filename)
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_training_data_cleaning_component(component_name, wait_for_completion=False):

    inputs, outputs = generic_component_inputs_outputs()
    run_azure_component(component_name, inputs, outputs, wait_for_completion)


def main():
    component_name = "training_data_cleaning"
    display_name = "Data cleaning for training"
    description = "Reads a .csv file and cleans duplicates and converts the labels"
    code_folder = "./src/pipeline_components/training_data_cleaning"
    code_filename = "training_data_cleaning_component.py"
    create_training_data_cleaning_component(component_name, display_name, description, 
                                            code_folder, code_filename)
    run_training_data_cleaning_component(component_name, wait_for_completion=False)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    
