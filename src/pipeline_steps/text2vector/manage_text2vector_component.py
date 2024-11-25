import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.azure_ml_interface import AzureMLInterface


def create_text2vector_component():
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
            "input_data_folder": Input(type="uri_folder"),
            "input_x_filename": Input(type="string"),
            "input_y_filename": Input(type="string"),
            "text2vec_data_filename": Input(type="string", default="text2vect_X_train.csv")
        },
        outputs=dict(
            model_path=Output(type="uri_file"),
            output_folder_path=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/text2vector",
        command="""python text2vector_component.py \
                --input_data_folder ${{inputs.input_data_folder}}\
                --input_x_filename ${{inputs.input_x_filename}} --input_y_filename ${{inputs.input_y_filename}} \
                --text2vec_data_filename $[[text2vect_X_train.csv]] \
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

    inputs = {
        "input_data_folder": folder_path,
        "input_x_filename": "X_train.csv",
        "input_y_filename": "y_train.csv"
    }
    outputs = {"model_path": model_path, "output_folder_path": output_folder_path}

    environment_variables = {"SCORING": os.getenv("SCORING", "f1_macro")}
    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion,
                                     environment_variables=environment_variables)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_text2vector_component()
