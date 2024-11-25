import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.azure_ml_interface import AzureMLInterface


def create_handle_imbalance_cleaning_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    handle_imbalance_cleaning_component = command(
        name="handle_imbalance",
        display_name="Uses techniques to overcome the imbalance labels",
        description=("Tests 4 methods to sample the data, chooses the best one and"
                     " returns the new sampled data."),
        inputs={
            "input_x": Input(type="uri_file"),
            "input_y": Input(type="uri_file"),
        },
        outputs=dict(
            model_path=Output(type="uri_file"),
            output_folder_path=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/handle_imbalance",
        command="""python handle_imbalance_component.py \
                --input_x ${{inputs.input_x}} --input_y ${{inputs.input_y}} \
                --model_path ${{outputs.model_path}} --output_folder_path ${{outputs.output_folder_path}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(handle_imbalance_cleaning_component)


def run_handle_imbalance_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "handle_imbalance"

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    folder_path = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                   f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    folder_path = "/".join(folder_path.split("/")[:-1]) 

    output_folder_path = Output(type="uri_folder", path=folder_path)
    model_path = os.path.join(folder_path, "best_imb_model.pickle")
    model_path = Output(type="uri_file", path=model_path)

    input_x = folder_path + "/text2vect_X_train.csv"
    input_y = folder_path + "/y_train.csv"

    inputs = {
        "input_x": input_x,
        "input_y": input_y
    }
    outputs = {"model_path": model_path, "output_folder_path": output_folder_path}

    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_handle_imbalance_cleaning_component()
