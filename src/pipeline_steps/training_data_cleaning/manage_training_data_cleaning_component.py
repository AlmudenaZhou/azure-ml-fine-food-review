import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.azure_ml_interface import AzureMLInterface


def create_training_data_cleaning_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    training_data_cleaning_component = command(
        name="training_data_cleaning",
        display_name="Generic data cleaning for training",
        description="reads a .csv file and cleans duplicates and converts the labels",
        inputs={
            "data": Input(type="uri_file"),
            "clean_filename": Input(type="string", optional=True, default="cleaned_data.csv"),
        },
        outputs=dict(
            clean_data=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/training_data_cleaning",
        command="""python training_data_cleaning_component.py \
                --data ${{inputs.data}} $[[--clean_filename ${{inputs.clean_filename}}]] \
                --clean_data ${{outputs.clean_data}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(training_data_cleaning_component)


def run_training_data_cleaning_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "training_data_cleaning"

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    data_uri = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    
    output_data_uri = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                       f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    output_data_uri = "/".join(output_data_uri.split("/")[:-1])
    clean_data_output = Output(type="uri_folder", path=output_data_uri)

    inputs = {
        "data": data_uri,
    }
    outputs = {"clean_data": clean_data_output}
    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_training_data_cleaning_component()
