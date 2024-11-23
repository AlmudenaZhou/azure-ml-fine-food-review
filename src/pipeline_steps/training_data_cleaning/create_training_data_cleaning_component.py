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
            "clean_filename": Input(type="string"),
        },
        outputs=dict(
            clean_data=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/training_data_cleaning",
        command="""python training_data_cleaning_component.py \
                --data ${{inputs.data}} --clean_filename ${{inputs.clean_filename}} \
                --clean_data ${{outputs.clean_data}}\
                """,


        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(training_data_cleaning_component)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_training_data_cleaning_component()