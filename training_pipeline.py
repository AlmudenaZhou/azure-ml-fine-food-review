import os

import logging.config

from dotenv import load_dotenv
import pandas as pd

from sklearn.model_selection import train_test_split

# from src.config import config
from azure_ml_interface import AzureMLInterface
from pipeline_steps.load_data_step import LoadDataStep
from src.pipeline_steps.handle_imbalance_step import HandleImbalanceStep
from src.pipeline_steps.text2vector_step import Text2VectorStep
from src.preprocessing.training_data_cleaning import data_to_binary, training_dataset_cleaning
from src.preprocessing.text_processing.processing_pipeline import preprocess_text


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)

load_dotenv()


class TrainingPipeline:

    def __init__(self):
        self.train_data_steps = [Text2VectorStep, HandleImbalanceStep]
        self.pospreprocess_data = os.getenv("POSPREPROCESS_DATA_FILE")
        if os.getenv("ENVIRONMENT") != "local":
            subscription_id = os.getenv("SUBSCRIPTION_ID")
            resource_group = os.getenv("RESOURCE_GROUP")
            workspace = os.getenv("WORKSPACE")
            self.azure_ml_interface = AzureMLInterface(subscription_id, resource_group, workspace)

    def _preprocessing_training_data(self, data):
        logger.info("Cleaning the dataset")
        data = training_dataset_cleaning(data)
        logger.info("Converting the label to binary")
        data = data_to_binary(data)        
        return data

    def _preprocessing_data(self, data):
        logger.info("Starting preprocess")
        data['Text'] = preprocess_text(data['Text'])
        data = data.loc[:, ['Text', 'Label']]
        return data
    
    def _save_posprocessed_data(self, data, prefix=""):
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        temp_dir = "./temp_data"
        os.makedirs(temp_dir, exist_ok=True)

        output_filename = os.environ["POSPROCESSED_TRAINING_DATA_FILENAME"]
        local_filepath = os.path.join(temp_dir, prefix + output_filename)
        data.to_parquet(local_filepath, index=False)

        if os.getenv("ENVIRONMENT") != "local":
            self.azure_ml_interface.create_urifile_dataasset_from_local_file(local_filepath, description=f"{prefix} posprocessed data", 
                                                                             name=f"{prefix}posprocessed", version="0")
            os.remove(local_filepath)

    def main(self, input_data_uri=None):
        logger.info("Starting training pipeline")
        data = self.pospreprocess_data
        if not data:
            logger.info("Starting all data steps...")
            data = LoadDataStep().main(input_data_uri)
            data = self._preprocessing_training_data(data)
            data = self._preprocessing_data(data)
            data.to_csv("preprocessed_data.csv")

        logger.info("Splitting the data...")
        test_size = float(os.environ["TEST_SIZE"])
        X_train, X_test, y_train, y_test = train_test_split(data.Text, data.Label, test_size=test_size)
        for step in self.steps:
            X_train, y_train = step().main(X_train, y_train)

        self._save_posprocessed_data(X_train, prefix="x_")
        self._save_posprocessed_data(y_train, prefix="y_")

    def train_model(self):
        pass


TrainingPipeline().main()
