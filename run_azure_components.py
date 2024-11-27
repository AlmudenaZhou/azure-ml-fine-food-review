import logging.config

from src.training_pipeline.model_training.manage_model_training_component import run_model_training_component
from src.training_pipeline.handle_imbalance.manage_handle_imbalance_component import run_handle_imbalance_component
from src.training_pipeline.text2vector.manage_text2vector_component import run_text2vector_component
from src.training_pipeline.split_data.manage_split_data_component import run_split_data_component
from src.training_pipeline.text_processing.manage_text_processing_component import run_text_processing_component
from src.training_pipeline.training_data_cleaning.manage_training_data_cleaning_component import run_training_data_cleaning_component
from src.tools.azure_ml_utils import manage_compute_instance_starting


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


def main():
    manage_compute_instance_starting()
    run_training_data_cleaning_component(wait_for_completion=True)
    run_text_processing_component(wait_for_completion=True)
    run_split_data_component(wait_for_completion=True)
    run_text2vector_component(wait_for_completion=True)
    run_handle_imbalance_component(wait_for_completion=True)
    run_model_training_component()


if __name__ == "__main__":
    main()