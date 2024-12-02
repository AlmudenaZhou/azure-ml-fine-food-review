from src.tools.azure_ml_utils import (create_azure_component, generic_creation_component_inputs_outputs,
                                      generic_running_component_inputs_outputs, run_azure_component)


def create_text_processing_component():
    component_name = "text_processing"
    display_name = "Text processing for training"
    description = "Get the text data and make sentence and tokens processing"
    code_folder = "./src/pipeline_components/text_processing"
    code_filename = "text_processing_component.py"
    inputs, outputs, code_command = generic_creation_component_inputs_outputs(code_filename)
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_text_processing_component(wait_for_completion=False):
    input_data_filename = "cleaned_data.csv"
    output_data_filename = "processed_data.csv"
    inputs, outputs = generic_running_component_inputs_outputs(input_data_filename, output_data_filename)
    run_azure_component("text_processing", inputs, outputs, wait_for_completion)


def main():
    create_text_processing_component()
    run_text_processing_component(wait_for_completion=False)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()