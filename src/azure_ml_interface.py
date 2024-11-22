import logging

from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import command
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ComputeInstance


logger = logging.getLogger(__name__)


class AzureMLInterface:

    def __init__(self, subscription_id, resource_group, workspace):
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

        # connect to workspace and submit job
        returned_job = self.ml_client.create_or_update(job)
        return returned_job

    def create_urifile_dataasset_from_local_file(self, local_filepath, description="", name="", version="0"):

        my_data = Data(
            path=local_filepath,
            type=AssetTypes.URI_FILE,
            description=description,
            name=name,
            version=version
        )

        self.ml_client.data.create_or_update(my_data)

    def create_compute_instance(self, ci_basic_name, ci_size="Standard_DS11_v2"):

        ci_basic = ComputeInstance(
            name=ci_basic_name, 
            size=ci_size
        )
        self.ml_client.begin_create_or_update(ci_basic).result()

    def create_environment_from_dockerfile(self,):
        env_docker_context = Environment(
            build=BuildContext(path="./"),
            name="docker-context-example",
            description="Environment created from a Docker context.",
        )
        self.ml_client.environments.create_or_update(env_docker_context)
