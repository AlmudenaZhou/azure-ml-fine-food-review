import os

from azure.ai.ml import Input, Output
from src.tools.azure_ml_utils import create_azure_component, run_azure_component


def create_split_data_component():
    component_name = "split_data"
    display_name = "Split data into train and test for training"
    description = "Split data in x_train, y_train, x_test, y_test and returns train_data and test_data"
    code_folder = "./src/pipeline_components/split_data"
    code_command = """python split_data_component.py \
        --input_data_folder ${{inputs.input_data_folder}}\
        --input_data_filename ${{inputs.input_data_filename}}\
        $[[--train_output_filename  ${{inputs.train_output_filename}}]] \
        $[[--test_output_filename  ${{inputs.test_output_filename}}]] \
        --output_data_folder ${{outputs.output_data_folder}} \
        """

    inputs = {
        "input_data_folder": Input(type="uri_folder"),
        "input_data_filename": Input(type="string"),
        "train_output_filename": Input(type="string", optional=True, default="train_data.csv"),
        "test_output_filename": Input(type="string", optional=True, default="test_data.csv"),
    }
    outputs = dict(
        output_data_folder=Output(type="uri_folder", mode="rw_mount")
    )
    create_azure_component(component_name, display_name, description, 
                            inputs, outputs, code_folder, code_command)


def run_split_data_component(wait_for_completion=False):

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    data_uri = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    
    data_folder, _ = os.path.split(data_uri)
    output_folder = Output(type="uri_folder", path=data_folder)

    inputs = {
        "input_data_folder": data_folder,
        "input_data_filename": "processed_data.csv",
        "train_output_filename": "train_data.csv",
        "test_output_filename": "test_data.csv"
    }
    outputs = {"output_data_folder": output_folder}

    environment_variables = {"TEST_SIZE": os.environ["TEST_SIZE"]}
    run_azure_component("split_data", inputs, outputs, wait_for_completion,
                        environment_variables=environment_variables)
    

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_split_data_component()
