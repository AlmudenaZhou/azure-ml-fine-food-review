import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.tools.azure_ml_interface import AzureMLInterface


def create_model_training_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    model_training_component = command(
        name="model_training",
        display_name="Simple model training",
        description=("Tests 3 types of models and chooses the best one. Not scalable to big amount of data."),
        inputs={
            "input_data_folder": Input(type="uri_folder"),
            "input_x_filename": Input(type="string"),
            "input_y_filename": Input(type="string"),
            "model_filename": Input(type="string", optional=True, default="model.pkl"),
        },
        outputs=dict(
            model_path=Output(type="uri_folder"),
            output_folder_path=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/training_pipeline/model_training",
        command="""python model_training_component.py \
                --input_data_folder ${{inputs.input_data_folder}}\
                --input_x_filename ${{inputs.input_x_filename}} --input_y_filename ${{inputs.input_y_filename}} \
                $[[--model_filename ${{inputs.model_filename}}]] \
                --model_path ${{outputs.model_path}} --output_folder_path ${{outputs.output_folder_path}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(model_training_component)


def run_model_training_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "model_training"

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    folder_path = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                   f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    folder_path = "/".join(folder_path.split("/")[:-1]) 

    output_folder_path = Output(type="uri_folder", path=folder_path)
    model_path = Output(type="uri_folder", path=folder_path)

    inputs = {
        "input_data_folder": folder_path,
        "input_x_filename": "imb_X_train.csv",
        "input_y_filename": "imb_y_train.csv",
        "model_filename": "model.pkl"
    }
    outputs = {"model_path": model_path, "output_folder_path": output_folder_path}

    environment_variables = {"SCORING": os.environ["SCORING"]}
    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion,
                                     environment_variables=environment_variables)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_model_training_component()
