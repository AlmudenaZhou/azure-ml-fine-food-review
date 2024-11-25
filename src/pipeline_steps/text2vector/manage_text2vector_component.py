import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.azure_ml_interface import AzureMLInterface


def create_text2vector_cleaning_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    text2vector_cleaning_component = command(
        name="text2vector",
        display_name="Text dataframe or series to vector using a model",
        description=("Gets a csv with X_train and y_train and converts the X_train to vector" 
         "and saves the model used."),
        inputs={
            "input_x": Input(type="uri_file"),
            "input_y": Input(type="uri_file"),
        },
        outputs=dict(
            model_path=Output(type="uri_file"),
            output_folder_path=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/text2vector",
        command="""python text2vector_component.py \
                --input_x ${{inputs.input_x}} --input_y ${{inputs.input_y}} \
                --model_path ${{outputs.model_path}} --output_folder_path ${{outputs.output_folder_path}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(text2vector_cleaning_component)


def run_text2vector_component(wait_for_completion=False):
    azure_ml_interface = AzureMLInterface()
    component_name = "text2vector"

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    folder_path = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                   f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    folder_path = "/".join(folder_path.split("/")[:-1]) 

    output_folder_path = Output(type="uri_folder", path=folder_path)
    model_path = os.path.join(folder_path, "best_text2vec_model.pickle")
    model_path = Output(type="uri_file", path=model_path)

    input_x = folder_path + "/X_train.csv"
    input_y = folder_path + "/y_train.csv"
    print("input_x", input_x)
    print("input_y", input_y)

    inputs = {
        "input_x": input_x,
        "input_y": input_y
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
    create_text2vector_cleaning_component()
