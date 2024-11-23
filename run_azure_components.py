import os
import logging.config

from src.azure_ml_interface import AzureMLInterface
from src.pipeline_steps.split_data.create_split_data_component import run_text_processing_component
from src.pipeline_steps.training_data_cleaning.create_training_data_cleaning_component import run_training_data_cleaning_component


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)

def main():
    azure_ml_interface = AzureMLInterface()
    compute_name = os.getenv("COMPUTE_INSTANCE_NAME")
    comp_status = azure_ml_interface.get_compute_status(compute_name)

    if comp_status == 'stopped':
        azure_ml_interface.start_compute()

    run_training_data_cleaning_component(wait_for_completion=True)
    run_text_processing_component(wait_for_completion=True)


if __name__ == "__main__":
    main()