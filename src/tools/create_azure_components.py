import logging

from src.pipeline_components.handle_imbalance.manage_handle_imbalance_component import create_handle_imbalance_component
from src.pipeline_components.split_data.manage_split_data_component import create_split_data_component
from src.pipeline_components.text2vector.manage_text2vector_component import create_text2vector_component
from src.pipeline_components.text_processing.manage_text_processing_component import create_text_processing_component
from src.pipeline_components.train_predict_model.manage_train_predict_model_component import create_train_predict_model_component
from src.pipeline_components.training_data_cleaning.manage_training_data_cleaning_component import create_training_data_cleaning_component


logger = logging.getLogger(__name__)

def create_components():

    create_text_processing_component()
    logger.info("Finished text processing component")

    create_split_data_component()
    logger.info("Finished split data component")

    create_text2vector_component()
    logger.info("Finished text2vector component")

    create_handle_imbalance_component()
    logger.info("handle imbalance component")

    create_train_predict_model_component()
    logger.info("Finished model training component")
    logger.info("Finished creating training components")
