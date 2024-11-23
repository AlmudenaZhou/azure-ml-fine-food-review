import logging.config
from dotenv import load_dotenv

from src.pipeline_steps.text_processing.create_text_processing_component import create_text_processing_component
from src.pipeline_steps.training_data_cleaning.create_training_data_cleaning_component import create_training_data_cleaning_component


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


def main():
    create_training_data_cleaning_component()
    create_text_processing_component()


if __name__ == "__main__":
    load_dotenv()
    main()
