import os

from azure.ai.ml import command
from azure.ai.ml import Input

from src.tools.azure_ml_interface import AzureMLInterface


def create_register_model_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    register_model_component = command(
        name="register_model",
        display_name="Simple model training",
        description=("Tests 3 types of models and chooses the best one. Not scalable to big amount of data."),
        inputs={
            "model_path": Input(type="uri_folder"),
        },
        outputs=dict(),
        code="./src/training_pipeline/register_model",
        command="""python register_model_component.py \
                --model_path ${{inputs.model_path}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(register_model_component)


def run_register_model_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "register_model"

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    folder_path = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                   f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    folder_path = "/".join(folder_path.split("/")[:-1]) 

    inputs = {
        "model_path": folder_path
    }
    outputs = {}

    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_register_model_component()
