import os
import sys
import pickle
import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(__file__))

from word2vec_model import Word2VecModel


logger = logging.getLogger(__name__)


class Text2VectorStep:

    def __init__(self, text_colname, target, model_path='models/best_text2vec.pickle'):

        self.text_colname = text_colname
        self.target = target
        self.model_path = model_path
    
    def main(self, train_data, is_training):
        if is_training:
            training_text2vec_step = TrainingText2VectorStep(self.text_colname, self.target, 
                                                             self.model_path)
            train_data = training_text2vec_step.main(train_data)
        else:
            inference_text2vec_step = InferenceText2VectorStep(self.text_colname, self.model_path)
            train_data = inference_text2vec_step.main(train_data)
        return train_data


class TrainingText2VectorStep:

    def __init__(self, text_colname, target, model_path):
        self.text_colname = text_colname
        self.target = target
        self.model_path = model_path

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
        scoring = os.getenv("SCORING", "f1_macro")
        for model_name, model in self.text2vec_models.items():
            with mlflow.start_run(run_name=f"{model_name}-run"):
                logger.info(f"Starting {model_name} training")
                dummy_model = self.dummy_model()
                new_x_train = model.fit_transform(X_train)
                mlflow.log_param("model_name", model_name)
                all_scores[model_name] = cross_val_score(dummy_model, new_x_train, y_train, cv=10, scoring=scoring)
                mlflow.log_param(f"scoring_metric", scoring)
                for fold_idx, score in enumerate(all_scores[model_name]):
                    mlflow.log_metric(f"{scoring}_fold_{fold_idx+1}", score)
                mean_score = all_scores[model_name].mean()
                mlflow.log_metric(f"{scoring}_mean", mean_score)

        all_scores = pd.DataFrame(all_scores).T
        best_model_name = all_scores.mean(axis=1).idxmax()
        logger.info(f"Text to Vec model chosen: {best_model_name}")
        best_text2vec = self.text2vec_models[best_model_name]
        return best_text2vec

    @staticmethod
    def save_model(best_text2vec, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(best_text2vec, file, protocol=pickle.HIGHEST_PROTOCOL)

    def main(self, train_data):
        
        X_train = train_data[self.text_colname]
        y_train = train_data[self.target]

        print(X_train.head())

        if self.best_text2vec is None:
            best_text2vec = self._choose_best_text_to_vector_model(X_train=X_train, y_train=y_train)
        X_train = best_text2vec.transform(X_train)
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.toarray()

        X_train = pd.DataFrame(X_train, index=y_train.index)
        logger.info("Resampled with the model")
        self.save_model(best_text2vec=best_text2vec, model_path=self.model_path)
        logger.info("Text to Vec model saved")

        train_data = pd.concat([X_train, y_train], axis=1)
        return train_data
    

class InferenceText2VectorStep:

    def __init__(self, text_colname, model_path):
        self.text_colname = text_colname
        self.model_path = model_path

    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        return model

    def main(self, train_data):

        data_x = train_data[self.text_colname]
        model = self.load_model(self.model_path)

        data_transf = model.transform(data_x)

        if not isinstance(data_transf, np.ndarray):
            data_transf = data_transf.toarray()

        data_transf = pd.DataFrame(data_transf, index=data_x.index)
        return data_transf
