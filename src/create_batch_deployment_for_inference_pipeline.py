import os
import time
import logging.config

from dotenv import load_dotenv

if __name__ == "__main__":
    import sys
    
    sys.path.append(os.getcwd())


from src.tools.azure_ml_interface import AzureMLInterface

logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


def main():

    azure_ml_interface = AzureMLInterface()

    compute_cluster_name = "batch-cluster"
    batch_endpoint_name = os.getenv("INFERENCE_BATCH_ENDPOINT_NAME")

    azure_ml_interface.create_compute_cluster(compute_name=compute_cluster_name)
    inference_pipeline = azure_ml_interface.get_component(component_name="fine_food_reviews_inference_pipeline")
    batch_endpoint = azure_ml_interface.create_batch_endpoint(batch_endpoint_name)
    deployment = azure_ml_interface.create_pipeline_component_batch_deployment(batch_endpoint_name, inference_pipeline, batch_endpoint.name, compute_cluster_name)
    azure_ml_interface.configure_deployment_as_default_in_endpoint(batch_endpoint.name, deployment.name)

    batch_endpoint = azure_ml_interface.get_batch_endpoint_from_name(batch_endpoint_name)

    time.sleep(50)
    print(f"The default deployment is {batch_endpoint.defaults.deployment_name}")


if __name__ == "__main__":
    load_dotenv()

    main()
