from abc import ABC, abstractmethod
import os
import pickle
from typing import Optional, Tuple
import logging.config

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# from src.config import config
from src.training_data_cleaning import data_to_binary, training_dataset_cleaning
from src.text_processing.processing_pipeline import preprocess_text
from src.text_processing.word2vec_model import Word2VecModel

from src.azure_ml_interface import AzureMLInterface


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)

load_dotenv()


class PipelineStep(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def main(self, X_train, y_train) -> Tuple[Optional[pd.DataFrame | pd.Series], Optional[pd.DataFrame| pd.Series]]:
        return X_train, y_train


class Text2VectorStep:

    def __init__(self):

        self.dummy_model = LogisticRegression
        
        self.text2vec_models = {'count_vec_ngram1': CountVectorizer(ngram_range=(1, 3), max_features=300),
                                'count_vec_ngram2': CountVectorizer(ngram_range=(1, 2), max_features=300),
                                'tfidf_ngram1': TfidfVectorizer(ngram_range=(1, 3), max_features=300),
                                'tfidf_ngram2': TfidfVectorizer(ngram_range=(1, 2), max_features=300),
                                'word2vec': Word2VecModel(min_count=1, window=10, vector_size=300, sample=6e-5,
                                                          alpha=0.03, min_alpha=0.0007, negative=20)}
        self.best_text2vec = None    

    def _choose_best_text_to_vector_model(self, X_train, y_train):
        all_scores = {}
        for model_name, model in self.text2vec_models.items():
            logging.info(f"Starting {model_name} training")
            dummy_model = self.dummy_model()
            new_x_train = model.fit_transform(X_train)
            all_scores[model_name] = cross_val_score(dummy_model, new_x_train, y_train, cv=10, scoring=os.environ["SCORING"])

        all_scores = pd.DataFrame(all_scores).T
        best_model_name = all_scores.mean(axis=1).idxmax()
        logging.info(f"Text to Vec model chosen: {best_model_name}")
        best_text2vec = self.text2vec_models[best_model_name]
        return best_text2vec

    @staticmethod
    def save_model(best_text2vec):
        if hasattr(best_text2vec, "save_model"):
            best_text2vec.save_model("best_text2vec.model")
        else:
            with open('best_text2vec.pickle', 'wb') as file:
                pickle.dump(best_text2vec, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def main(self, X_train, y_train):

        if self.best_text2vec is None:
            best_text2vec = self._choose_best_text_to_vector_model(X_train=X_train, y_train=y_train)
        X_train = best_text2vec.transform(X_train)
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.toarray()

        X_train = pd.DataFrame(X_train, index=y_train.index)
        logging.info("Resampled with the model")
        self.save_model(best_text2vec=best_text2vec)
        logging.info("Text to Vec model saved")
        return X_train, y_train


class HandleImbalanceStep:

    def __init__(self):
        self.dummy_model = LogisticRegression

        self.imb_models = {'base': "base",
                           'undersampling': RandomUnderSampler(),
                           'oversampling': RandomOverSampler(),
                           'smote': SMOTE()}
        self.best_imb_model = None

    def _individual_imb_model_train(self, model, X_train_orig, y_train_orig):
        ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        ind_results = []
        for train_index, test_index in ss.split(X_train_orig.values):
            x_train, y_train = X_train_orig.iloc[train_index], y_train_orig.iloc[train_index]
            x_test, y_test = X_train_orig.iloc[test_index], y_train_orig.iloc[test_index]

            x_train_imb, y_train_imb = x_train, y_train
            if model != "base":
                x_train_imb, y_train_imb = model.fit_resample(x_train, y_train)

            pred_model = self.dummy_model()
            pred_model.fit(x_train_imb, y_train_imb)
            y_pred = pred_model.predict(x_test)
            ind_results.append(f1_score(y_test, y_pred, average='macro'))
        return ind_results

    def _choose_best_imb_model(self, X_train_orig, y_train_orig):
        all_results = {}
        for model_name, model in self.imb_models.items():
            logging.info(f"Starting {model_name} training")
            ind_results = self._individual_imb_model_train(model, X_train_orig, y_train_orig)
            all_results[model_name] = ind_results

        all_results = pd.DataFrame(all_results).T
        best_model_name = all_results.mean(axis=1).idxmax()
        logging.info(f"Imbalanced model chosen: {best_model_name}")
        return self.imb_models[best_model_name]
    
    @staticmethod
    def save_model(best_imb_model):
        with open('best_imb_model.pickle', 'wb') as file:
            pickle.dump(best_imb_model, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def main(self, X_train, y_train):
        train_columns = X_train.columns
        index = y_train.index
        if self.best_imb_model is None:
            self.best_imb_model = self._choose_best_imb_model(X_train, y_train)

        if self.best_imb_model != "base":
            X_train, y_train = self.best_imb_model.fit_resample(X_train, y_train)
            logging.info("Resampled with the model")
            self.save_model(self.best_imb_model)
            logging.info("Imb model saved")

        X_train = pd.DataFrame(X_train, index=index, columns=train_columns)
        return X_train, y_train


class TrainingPipeline:

    def __init__(self):

        relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")
        if os.getenv("ENVIRONMENT") == "local":
            filename = relative_raw_data_uri.split('/')[-1]
            self.raw_data_uri = os.path.join("data", filename)
        else:
            subscription_id = os.getenv("SUBSCRIPTION_ID")
            resource_group = os.getenv("RESOURCE_GROUP")
            workspace = os.getenv("WORKSPACE")

            self.raw_data_uri = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                                f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
            
            self.azure_ml_interface = AzureMLInterface(subscription_id, resource_group, workspace)
        
        self.steps = [Text2VectorStep, HandleImbalanceStep]
        self.pospreprocess_data = os.getenv("POSPREPROCESS_DATA_FILE")

    def _preprocessing_data(self):
        logging.info(f"Reading {self.raw_data_uri}")
        raw_data = pd.read_csv(self.raw_data_uri)
        logging.info("Cleaning the dataset")
        data = training_dataset_cleaning(raw_data)
        logging.info("Converting the label to binary")
        data = data_to_binary(data)
        logging.info("Starting preprocess")
        data['Text'] = preprocess_text(data['Text'])
        data = data.loc[:, ['Text', 'Label']]
        return data
    
    def _save_posprocessed_data(self, data, prefix=""):
        
        temp_dir = "./temp_data"
        os.makedirs(temp_dir, exist_ok=True)

        output_filename = os.environ["POSPROCESSED_TRAINING_DATA_FILENAME"]
        local_filepath = os.path.join(temp_dir, prefix + output_filename)
        data.to_parquet(local_filepath, index=False)

        if os.getenv("ENVIRONMENT") != "local":
            self.azure_ml_interface.create_urifile_dataasset_from_local_file(local_filepath, description=f"{prefix} posprocessed data", 
                                                                             name=f"{prefix}posprocessed", version="0")

    def main(self):
        logging.info("Starting training pipeline")
        data = self.pospreprocess_data
        if not data:
            logging.info("Starting data preprocessing...")
            data = self._preprocessing_data()
            data.to_csv("preprocessed_data.csv")

        logging.info("Splitting the data...")
        test_size = float(os.environ["TEST_SIZE"])
        X_train, X_test, y_train, y_test = train_test_split(data.Text, data.Label, test_size=test_size)
        for step in self.steps:
            X_train, y_train = step().main(X_train, y_train)

        self._save_posprocessed_data(X_train, prefix="x_")
        self._save_posprocessed_data(y_train, prefix="y_")

    def train_model(self):
        pass


TrainingPipeline().main()
