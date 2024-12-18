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
from azure.ai.ml.entities import Data, Model, AmlCompute, BatchEndpoint, PipelineComponentBatchDeployment
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
                          compute="aml-cluster", environment_variables=None):

        job = command(
            code=code_folder,
            command=command_line,
            environment=environment,
            compute=compute,
            experiment_name=experiment_name,
            environment_variables=environment_variables
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

        created_data = self.ml_client.data.create_or_update(my_data)
        logger.info("Data Asset created with ID: %s", created_data.id)

    def register_model_from_file(self, filepath, model_type=AssetTypes.CUSTOM_MODEL, description="Model created from local file.",
                                 name="", version=0):
        file_model = Model(
            path=filepath,
            type=model_type,
            name=name,
            description=description,
            version=version
        )
        model = self.ml_client.models.create_or_update(file_model)
        return model

    def create_compute_instance(self, ci_basic_name, ci_size="Standard_DS11_v2"):
        logger.info("Creating Compute Instance: ", ci_basic_name)
        result = None
        if not any(filter(lambda m: m.name == ci_basic_name, self.ml_client.compute.list())):
            ci_basic = ComputeInstance(
                name=ci_basic_name, 
                size=ci_size
            )
            result = self.ml_client.begin_create_or_update(ci_basic).result()
        return result

    def create_compute_cluster(self, compute_name=None, min_instances=0, max_instances=5, 
                               description="Batch endpoints compute cluster"):
        compute_name = compute_name if compute_name else "batch-cluster"
        result = None
        if not any(filter(lambda m: m.name == compute_name, self.ml_client.compute.list())):
            compute_cluster = AmlCompute(
                name=compute_name,
                description=description,
                min_instances=min_instances,
                max_instances=max_instances,
            )
            result = self.ml_client.begin_create_or_update(compute_cluster).result()
        return result
    
    def create_batch_endpoint(self, endpoint_name):
        endpoint = BatchEndpoint(
            name=endpoint_name,
            description="Batch scoring endpoint of the Heart Disease Data Set prediction task",
        )

        return self.ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    def create_pipeline_component_batch_deployment(self, deployment_name, pipeline_component, endpoint_name, default_compute_name):
        deployment = PipelineComponentBatchDeployment(
            name=deployment_name,
            description="Deployment from a pipeline component.",
            endpoint_name=endpoint_name,
            component=pipeline_component,
            settings={"continue_on_step_failure": False, "default_compute": default_compute_name},
        )

        return self.ml_client.batch_deployments.begin_create_or_update(deployment).result()
    
    def configure_deployment_as_default_in_endpoint(self, endpoint_name, deployment_name):
        endpoint = self.ml_client.batch_endpoints.get(endpoint_name)
        endpoint.defaults.deployment_name = deployment_name
        return self.ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    def get_batch_endpoint_from_name(self, endpoint_name):
        endpoint = self.ml_client.batch_endpoints.get(name=endpoint_name)
        return endpoint

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
                description="Environment created to run the Pipelines for Fine Food Reviews."
            )
            job_env = self.ml_client.environments.create_or_update(env_docker_context)

        logger.info(
            f"Environment with name {job_env.name} is registered to workspace, the environment version is {job_env.version}"
        )

    def create_component_from_component(self, component):
        component = self.ml_client.create_or_update(component.component)
        print(
            f"Component {component.name} with Version {component.version} is registered"
        )
    
    @staticmethod
    def generate_job_name(base_name="job"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # Shortened UUID for readability
        return f"{base_name}_{timestamp}_{unique_id}"

    def get_component(self, component_name, component_version=None):
        return self.ml_client.components.get(name=component_name, version=component_version,)

    def run_component(self, component_name, inputs, outputs=None, compute_instance=os.getenv("COMPUTE_INSTANCE_NAME"),
                      component_version=None, wait_for_completion=False, environment_variables=None):

        component = self.get_component(component_name, component_version)
        name = self.generate_job_name(base_name="job")

        job = command(
            name=name,
            component=component,
            inputs=inputs,
            outputs=outputs,
            compute=compute_instance,
            environment=component.environment,
            environment_variables=environment_variables
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

    def register_pipeline_from_job_pipeline(self, pipeline_job, environment_variables=None):
        registered_pipeline = self.ml_client.jobs.create_or_update(pipeline_job, 
                                                                   environment_variables=environment_variables)
        logger.info(f"Pipeline registered with name: {registered_pipeline.name}")
        return registered_pipeline

    def run_pipeline_by_pipeline_name(self, pipeline_name):
        self.ml_client.jobs.stream(pipeline_name)
        logger.info(f"Run pipeline with name: {pipeline_name}")
