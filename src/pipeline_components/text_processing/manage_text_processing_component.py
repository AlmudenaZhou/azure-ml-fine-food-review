import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.tools.azure_ml_interface import AzureMLInterface


def create_text_processing_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    text_processing_component = command(
        name="text_processing",
        display_name="Text processing for training",
        description="Get the text data and make sentence and tokens processing",
        inputs={
            "input_data_folder": Input(type="uri_folder"),
            "input_data_filename": Input(type="string"),
            "processed_filename": Input(type="string", optional=True, default="processed_data.csv"),
        },
        outputs=dict(
            processed_data=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/training_pipeline/text_processing",
        command="""python text_processing_component.py \
                --input_data_folder ${{inputs.input_data_folder}} --input_data_filename ${{inputs.input_data_filename}}\
                $[[--processed_filename ${{inputs.processed_filename}}]] \
                --processed_data ${{outputs.processed_data}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(text_processing_component)


def run_text_processing_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "text_processing"

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    folder_path = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                   f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    folder_path = "/".join(folder_path.split("/")[:-1]) 

    data_output = Output(type="uri_folder", path=folder_path)
    inputs = {
        "input_data_folder": folder_path,
        "input_data_filename": "cleaned_data.csv",
    }
    outputs = {
        "processed_data": data_output
    }

    azure_ml_interface.run_component(component_name=component_name, 
                                     inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_text_processing_component()