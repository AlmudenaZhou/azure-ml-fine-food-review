import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.azure_ml_interface import AzureMLInterface


def create_split_data_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    split_data_component = command(
        name="split_data",
        display_name="Split data for training",
        description="Split data in x_train, y_train, x_test, y_test",
        inputs={
            "input_data": Input(type="uri_file")
        },
        outputs=dict(
            split_data=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/text_processing",
        command="""python text_processing_component.py \
                --input_data ${{inputs.input_data}} \
                --split_data ${{outputs.split_data}} \
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(split_data_component)


def run_split_data_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "split_data"
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    folder_path = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                   f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    folder_path = "/".join(folder_path.split("/")[:-1]) 

    data_output = Output(type="uri_folder", path=folder_path)
    inputs = {
        "input_data": folder_path + "/preprocessed_data.csv",
    }
    outputs = {
        "processed_data": data_output
    }
    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_split_data_component()
