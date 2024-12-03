import os

import logging.config

from dotenv import load_dotenv
import pandas as pd

from src.tools.azure_ml_interface import AzureMLInterface
from src.pipeline_components.split_data.split_data_step import SplitDataStep
from src.pipeline_components.train_predict_model.train_predict_model_step import TrainPredictStep
from src.local_components.load_data.load_data_step import LoadDataStep
from src.pipeline_components.handle_imbalance.handle_imbalance_step import HandleImbalanceStep
from src.pipeline_components.text2vector.text2vector_step import Text2VectorStep
from src.pipeline_components.training_data_cleaning.training_data_cleaning_step import TrainingDataCleaningStep
from src.pipeline_components.text_processing.text_processing_step import TextPreprocessingStep
from src.local_components.register_model.register_model_step import RegisterModelStep


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)

load_dotenv()


class TrainingPipeline:

    def __init__(self):
        self.train_data_steps = [Text2VectorStep, HandleImbalanceStep]
        self.pospreprocess_data = os.getenv("POSPREPROCESS_DATA_FILE")
        self.text_colname = os.getenv("TEXT_COLNAME")
        self.target = os.getenv("TARGET")
        if os.getenv("ENVIRONMENT") != "local":
            subscription_id = os.getenv("SUBSCRIPTION_ID")
            resource_group = os.getenv("RESOURCE_GROUP")
            workspace = os.getenv("WORKSPACE")
            self.azure_ml_interface = AzureMLInterface(subscription_id, resource_group, workspace)
    
    def _save_posprocessed_data(self, data, prefix=""):
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        temp_dir = "./data"
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
            data = LoadDataStep(input_data_uri).main()
            data = TrainingDataCleaningStep().main(data)
            data = TextPreprocessingStep(self.text_colname).main(data)
            data.to_csv("data/preprocessed_data.csv")

        train_data, test_data = SplitDataStep(self.target).main(data)
        text2vec_step = Text2VectorStep(self.text_colname, self.target, "models/text2vec_model.pkl")
        train_data = text2vec_step.main(train_data, True)

        handle_imb_step = HandleImbalanceStep(self.target)
        train_data = handle_imb_step.main(train_data)

        self._save_posprocessed_data(train_data, prefix="")

        logger.info("Starting training...")
        model_training_step = TrainPredictStep(self.target, "models/predictor.pkl")
        model_training_step.main(train_data, is_training=True)
        logger.info("Best model trained")

        logger.info("Registering the models...")

        model_params_dict = {
            "text2vec": {
                 "model_path": text2vec_step.model_path, 
                 "model_name": "text2vec",
                 "model_version": "1",
                 "model_type": "custom_model"
                 },
            "predictor": {
                "model_path": model_training_step.model_path, 
                "model_name": "model_training",
                "model_version": "1",
                "model_type": "custom_model"
            }
        }
        models = RegisterModelStep().main(model_params_dict)
        logger.info("Finished Training Pipeline")


TrainingPipeline().main()
