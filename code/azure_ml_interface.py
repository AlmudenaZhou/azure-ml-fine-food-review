from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ComputeInstance


class AzureMLInterface:

    def __init__(self, subscription_id, resource_group, workspace):
        self.ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace
        )
    
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
