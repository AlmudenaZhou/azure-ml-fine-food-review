import os

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.tools.azure_ml_interface import AzureMLInterface


def create_text2vector_inference_component():
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    text2vector_component = command(
        name="text2vector_inference",
        display_name="Text dataframe or series to vector using a model",
        description=("Gets the data transformed by the model"),
        inputs={
            "model": Input(type="custom_model"),
            "input_data_folder": Input(type="uri_folder"),
            "input_data_filename": Input(type="string"),
            "output_data_filename": Input(type="string"),
        },
        outputs=dict(
            output_data_folder=Output(type="uri_folder")
        ),
        code="./src/inference_pipeline/text2vector",
        command="""python text2vector_component.py \
                --model ${{inputs.model}}\
                --input_data_folder ${{inputs.input_data_folder}} \
                --input_data_filename ${{inputs.input_data_filename}} \
                --output_data_filename ${{inputs.output_data_filename}}\
                --output_data_folder ${{outputs.output_data_folder}}\
                """,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(text2vector_component)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_text2vector_inference_component()
