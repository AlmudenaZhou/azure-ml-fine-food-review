import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.azure_ml_interface import AzureMLInterface


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
            "input_data": Input(type="uri_file"),
            "processed_filename": Input(type="string"),
        },
        outputs=dict(
            processed_data=Output(type="uri_folder", mode="rw_mount")
        ),
        code="./src/pipeline_steps/text_processing",
        command="""python text_processing_component.py \
                --input_data ${{inputs.input_data}} --processed_filename ${{inputs.processed_filename}} \
                --processed_data ${{outputs.processed_data}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(text_processing_component)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_text_processing_component()