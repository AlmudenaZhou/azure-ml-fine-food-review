import os
import logging.config

from src.pipeline_steps.split_data.manage_split_data_component import run_split_data_component
from src.pipeline_steps.text_processing.manage_text_processing_component import run_text_processing_component
from src.azure_ml_interface import AzureMLInterface
from src.pipeline_steps.training_data_cleaning.manage_training_data_cleaning_component import run_training_data_cleaning_component


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)

def main():
    azure_ml_interface = AzureMLInterface()
    compute_name = os.getenv("COMPUTE_INSTANCE_NAME")
    comp_status = azure_ml_interface.get_compute_status(compute_name)
    logger.info("Compute instance Status:", comp_status.state)
    if comp_status.state == 'Stopped':
        azure_ml_interface.start_compute(compute_name)
        while comp_status != "Running":
            comp_status = azure_ml_interface.get_compute_status(compute_name)

    run_training_data_cleaning_component(wait_for_completion=True)
    run_text_processing_component(wait_for_completion=True)
    run_split_data_component(wait_for_completion=True)


if __name__ == "__main__":
    main()