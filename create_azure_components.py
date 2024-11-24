import logging.config
from dotenv import load_dotenv

from src.pipeline_steps.split_data.manage_split_data_component import create_split_data_component
from src.pipeline_steps.text_processing.manage_text_processing_component import create_text_processing_component
from src.pipeline_steps.training_data_cleaning.manage_training_data_cleaning_component import create_training_data_cleaning_component


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


def main():
    create_training_data_cleaning_component()
    create_text_processing_component()
    create_split_data_component()


if __name__ == "__main__":
    load_dotenv()
    main()
