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
            "input_data_folder": Input(type="uri_folder"),
            "input_data_filename": Input(type="string"),
            "x_train_filename": Input(type="string", optional=True, default="X_train.csv"),
            "x_test_filename": Input(type="string", optional=True, default="X_test.csv"),
            "y_train_filename": Input(type="string", optional=True, default="y_train.csv"),
            "y_test_filename": Input(type="string", optional=True, default="y_test.csv"),
        },
        outputs=dict(
            split_data=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/split_data",
        command="""python split_data_component.py \
                --input_data_folder ${{inputs.input_data_folder}}\
                --input_data_filename ${{inputs.input_data_filename}}\
                --x_train_filename $[[X_train.csv]] \
                --x_test_filename $[[X_test.csv]] \
                --y_train_filename $[[y_train.csv]] \
                --y_test_filename $[[y_test.csv]] \
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
        "input_data_folder": folder_path,
        "input_data_filename": "processed_data.csv",
    }
    outputs = {
        "split_data": data_output
    }
    environment_variables = {"TEST_SIZE": os.environ["TEST_SIZE"]}
    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion,
                                     environment_variables=environment_variables)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_split_data_component()
