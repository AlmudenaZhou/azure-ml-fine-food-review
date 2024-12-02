from src.tools.azure_ml_utils import (create_azure_component, generic_creation_component_inputs_outputs,
                                      generic_running_component_inputs_outputs, run_azure_component)


def create_handle_imbalance_component():

    component_name = "handle_imbalance"
    display_name = "Sampling data to handle imbalance labels"
    description = ("Tests 4 methods to sample the data, chooses the best one and"
                   " returns the new sampled data.")
    code_folder = "./src/pipeline_components/handle_imbalance"
    code_filename = "handle_imbalance_component.py"
    inputs, outputs, code_command = generic_creation_component_inputs_outputs(code_filename, is_model=True)
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_handle_imbalance_component(wait_for_completion=False):
    input_data_filename = "text2vector_data.csv"
    output_data_filename = "handled_imb_data.csv"
    model_filename = "imb_model.pkl"
    inputs, outputs = generic_running_component_inputs_outputs(input_data_filename, output_data_filename,
                                                               model_filename, is_model=True)
    run_azure_component("handle_imbalance", inputs, outputs, wait_for_completion)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_handle_imbalance_component()
