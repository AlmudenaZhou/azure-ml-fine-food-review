import os
import logging
import time

from azure.ai.ml import command
from azure.ai.ml import Input, Output

from src.tools.azure_ml_interface import AzureMLInterface


logger = logging.getLogger(__name__)

def manage_compute_instance_starting():

    azure_ml_interface = AzureMLInterface()

    compute_name = os.getenv("COMPUTE_INSTANCE_NAME")
    comp_status = azure_ml_interface.get_compute_status(compute_name)
    logger.info("Compute instance Status:", comp_status.state)

    if comp_status.state == 'Updating':
        while comp_status.state not in ['Stopped', 'Running']:
            comp_status = azure_ml_interface.get_compute_status(compute_name)
            time.sleep(2)

    if comp_status.state == 'Stopped':
        azure_ml_interface.start_compute(compute_name)
        while comp_status.state != "Running":
            comp_status = azure_ml_interface.get_compute_status(compute_name)
            time.sleep(2)


def create_azure_component(component_name, display_name, description, 
                           inputs, outputs, code_folder, code_command):
    azure_ml_interface = AzureMLInterface()

    env_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    envs = azure_ml_interface.ml_client.environments.list(name=env_name)
    for env in envs:
        env_version = env.version
        break

    training_data_cleaning_component = command(
        name=component_name,
        display_name=display_name,
        description=description,
        inputs=inputs,
        outputs=outputs,
        code=code_folder,
        command=code_command,
        environment=f'{env_name}:{env_version}',
    )

    print("Environment used: ", f'{env_name}:{env_version}')
    azure_ml_interface.create_component_from_component(training_data_cleaning_component)


def generic_creation_component_inputs_outputs(code_filename, is_model=False):
    inputs = {
        "input_data_folder": Input(type="uri_folder"),
        "input_data_filename": Input(type="string"),
        "output_data_filename": Input(type="string", optional=True, default="data.csv"),
    }

    outputs = dict(
        output_data_folder=Output(type="uri_folder", mode="rw_mount")
    )

    code_command = (f"python {code_filename} " + 
        """--input_data_filename ${{inputs.input_data_filename}}\
        --input_data_folder ${{inputs.input_data_folder}}\
        $[[--output_data_filename ${{inputs.output_data_filename}}]]\
        --output_data_folder ${{outputs.output_data_folder}}\
        """)

    if is_model:
        inputs["model_filename"] = Input(type="string", default="model.pkl",
                                         optional=True)
        inputs["is_training"] = Input(type="string")
        inputs["model_input_path"] = Input(type="custom_model", optional=True)
        outputs["model_output_folder"] = Output(type="uri_folder", mode="rw_mount")

        code_command += """--model_output_folder ${{outputs.model_output_folder}}\
            $[[--model_filename ${{inputs.model_filename}}]]\
            --is_training ${{inputs.is_training}}\
            $[[--model_input_path ${{inputs.model_input_path}}]]\
            """

    return inputs, outputs, code_command


def run_azure_component(component_name, inputs, outputs, wait_for_completion=False,
                        environment_variables=None):
    azure_ml_interface = AzureMLInterface()

    azure_ml_interface.run_component(component_name=component_name, inputs=inputs, outputs=outputs,
                                     compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                                     component_version=None, wait_for_completion=wait_for_completion,
                                     environment_variables=environment_variables)
    

def generic_running_component_inputs_outputs(input_data_filename, output_data_filename, model_filename="model.pkl",
                                             is_model=False, is_training="True", is_inference=False):
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    data_uri = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")

    data_folder, _ = os.path.split(data_uri)
    output_folder = Output(type="uri_folder", path=data_folder)

    inputs = {
        "input_data_folder": Input(type="uri_folder", path=data_folder),
        "input_data_filename": input_data_filename,
        "output_data_filename": output_data_filename,
    }
    outputs = {"output_data_folder": output_folder}

    if is_model:
        inputs['model_filename'] = model_filename
        inputs['is_training'] = is_training
        inputs['model_input_folder'] = Input(type="uri_folder", path=data_folder)
        outputs['model_output_folder'] = Output(type="uri_folder", path=data_folder)

    return inputs, outputs
