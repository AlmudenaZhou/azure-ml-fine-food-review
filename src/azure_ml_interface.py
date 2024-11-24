import datetime
import os
import time

import logging
import shutil
from tempfile import TemporaryDirectory
import uuid

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ComputeInstance
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import command
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ComputeInstance


logger = logging.getLogger(__name__)


class AzureMLInterface:

    def __init__(self, subscription_id=os.getenv("SUBSCRIPTION_ID"),
                 resource_group = os.getenv("RESOURCE_GROUP"),
                 workspace = os.getenv("WORKSPACE")):

        credential = self.get_credentials()
        self.ml_client = MLClient(
            credential, subscription_id, resource_group, workspace
        )

    @staticmethod
    def get_credentials():
        try:
            credential = DefaultAzureCredential()
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            credential = InteractiveBrowserCredential()
            logger.warning(f"Exception with the DefaultAzureCredentials: {ex}")
        return credential

    def run_a_command_job(self, experiment_name, code_folder="./src", command_line="python train.py",
                          environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
                          compute="aml-cluster",):
        job = command(
            code=code_folder,
            command=command_line,
            environment=environment,
            compute=compute,
            experiment_name=experiment_name
        )
        logger.info("Launching the command")
        # connect to workspace and submit job
        returned_job = self.ml_client.create_or_update(job)
        logger.info(f"Returned job: {returned_job}")
        return returned_job

    def create_urifile_dataasset_from_local_file(self, local_filepath, description="", name="", version="0"):
        logger.info("Creating the Data Asset from: ", local_filepath)
        my_data = Data(
            path=local_filepath,
            type=AssetTypes.URI_FILE,
            description=description,
            name=name,
            version=version
        )

        self.ml_client.data.create_or_update(my_data)

    def create_compute_instance(self, ci_basic_name, ci_size="Standard_DS11_v2"):
        logger.info("Creating Compute Instance: ", ci_basic_name)
        ci_basic = ComputeInstance(
            name=ci_basic_name, 
            size=ci_size
        )
        self.ml_client.begin_create_or_update(ci_basic).result()

    def get_compute_status(self, ci_basic_name):
        logger.info("Getting the status from Compute Instance: ", ci_basic_name)
        ci_basic_state = self.ml_client.compute.get(ci_basic_name)
        logger.info("Status: ", ci_basic_state)
        return ci_basic_state
    
    def stop_compute(self, ci_basic_name):
        logger.info("Stopping Compute Instance: ", ci_basic_name)
        self.ml_client.compute.begin_stop(ci_basic_name).wait()
    
    def start_compute(self, ci_basic_name):
        logger.info("Starting Compute Instance: ", ci_basic_name)
        self.ml_client.compute.begin_start(ci_basic_name).wait()

    def restart_compute(self, ci_basic_name):
        logger.info("Restarting Compute Instance: ", ci_basic_name)
        self.ml_client.compute.begin_restart(ci_basic_name).wait()

    def delete_compute(self, ci_basic_name):
        logger.info("Deleting Compute Instance: ", ci_basic_name)
        self.ml_client.compute.begin_delete(ci_basic_name).wait()

    def create_environment_from_dockerfile(self, environment_name=os.getenv("AZURE_ML_ENVIRONMENT_NAME")):

        with TemporaryDirectory() as temp_dir:
            shutil.copy("./Dockerfile", os.path.join(temp_dir, "Dockerfile"))
            shutil.copy("./requirements.txt", os.path.join(temp_dir, "requirements.txt"))

            env_docker_context = Environment(
                build=BuildContext(path=temp_dir),
                name=environment_name,
                description="Environment created to run the Pipelines for Fine Food Reviews.",
            )
            pipeline_job_env = self.ml_client.environments.create_or_update(env_docker_context)

        logger.info(
            f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
        )

    def create_component_from_component(self, component):
        component = self.ml_client.create_or_update(component.component)
        print(
            f"Component {component.name} with Version {component.version} is registered"
        )

    def run_component(self, component_name, inputs, outputs=None, compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                      component_version=None, wait_for_completion=False):
        
        def generate_job_name(base_name="job"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = uuid.uuid4().hex[:8]  # Shortened UUID for readability
            return f"{base_name}_{timestamp}_{unique_id}"

        component = self.ml_client.components.get(name=component_name, version=component_version)
        name = generate_job_name(base_name="job")
        job = command(
            name=name,
            component=component,
            inputs=inputs,
            outputs=outputs,
            compute=compute_instance,
            environment=component.environment
        )
        returned_job = self.ml_client.jobs.create_or_update(job)

        if wait_for_completion:
            status = returned_job.status
            i = 0
            while (status not in ["CancelRequested", "Completed", "Failed", "Canceled", "NotResponding"]):
                returned_job = self.ml_client.jobs.get(name=name)
                status = returned_job.status
                time.sleep(1)
                i += 1
                logger.debug(f"Status at second {i}: {status}")
