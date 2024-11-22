import pickle
import logging

import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


logger = logging.getLogger(__name__)

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
            logger.info(f"Starting {model_name} training")
            ind_results = self._individual_imb_model_train(model, X_train_orig, y_train_orig)
            all_results[model_name] = ind_results

        all_results = pd.DataFrame(all_results).T
        best_model_name = all_results.mean(axis=1).idxmax()
        logger.info(f"Imbalanced model chosen: {best_model_name}")
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
            logger.info("Resampled with the model")
            self.save_model(self.best_imb_model)
            logger.info("Imb model saved")

        X_train = pd.DataFrame(X_train, index=index, columns=train_columns)
        return X_train, y_train