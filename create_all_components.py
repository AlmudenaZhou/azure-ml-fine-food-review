import logging.config
from dotenv import load_dotenv

from src.inference_pipeline.model_prediction.manage_model_prediction_component import create_model_prediction_inference_component
from src.inference_pipeline.text2vector.manage_text2vector_component import create_text2vector_inference_component
from src.inference_pipeline.text_processing.manage_text_processing_component import create_text_processing_inference_component
from src.training_pipeline.model_training.manage_model_training_component import create_model_training_component
from src.training_pipeline.handle_imbalance.manage_handle_imbalance_component import create_handle_imbalance_component
from src.training_pipeline.text2vector.manage_text2vector_component import create_text2vector_component
from src.training_pipeline.split_data.manage_split_data_component import create_split_data_component
from src.training_pipeline.text_processing.manage_text_processing_component import create_text_processing_component
from src.training_pipeline.training_data_cleaning.manage_training_data_cleaning_component import create_training_data_cleaning_component


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


def create_all_components():
    logger.info("Stating creating all components")
    create_training_components()

    create_inference_components()


def create_training_components():
    logger.info("Stating creating training components")
    create_training_data_cleaning_component()
    logger.info("Finished training data cleaning component")

    create_text_processing_component()
    logger.info("Finished text processing component")

    create_split_data_component()
    logger.info("Finished split data component")

    create_text2vector_component()
    logger.info("Finished text2vector component")

    create_handle_imbalance_component()
    logger.info("handle imbalance component")

    create_model_training_component()
    logger.info("Finished model training component")
    logger.info("Finished creating training components")


def create_inference_components():

    create_text_processing_inference_component()

    create_text2vector_inference_component()

    create_model_prediction_inference_component()


def main():
    create_all_components()



if __name__ == "__main__":
    load_dotenv()
    main()
