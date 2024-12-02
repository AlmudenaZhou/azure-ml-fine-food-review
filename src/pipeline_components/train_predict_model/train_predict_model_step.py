import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import mlflow


class TrainPredictStep:

    def __init__(self, target, model_path):
        self.target = target
        self.model_path = model_path

    def main(self, train_data, is_training):
        if is_training:
            model_training_step = ModelTrainingStep(self.target, self.model_path)
            preds = model_training_step.main(train_data)
        else:
            model_pred_step = ModelPredictionStep(self.target, self.model_path)
            preds = model_pred_step.main(train_data)
        return preds


class ModelTrainingStep:

    def __init__(self, target, model_path='models/best_model.pickle'):
        self.target = target
        self.model_dict = {'logreg': LogisticRegression(),
                           'SVC': SVC(),
                           'dectree': DecisionTreeClassifier()}
        self.best_model = None
        self.model_path = model_path
        
    def _choose_best_model(self, X_train, y_train,
                           scoring='f1_macro', cv=10):
        mlflow.autolog()
        with mlflow.start_run():
            all_scores = [cross_val_score(model, X_train, y_train,
                                          cv=cv, scoring=scoring)
                        for model in self.model_dict.values()]
            all_results = pd.DataFrame(all_scores, index=self.model_dict.keys())
            best_model_name = all_results.mean(axis=1).idxmax()
            best_model = self.model_dict[best_model_name]
        return best_model

    def _train_best_model(self, X_train, y_train):
        self.best_model.fit(X_train, y_train)

    @staticmethod
    def save_model(best_model, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(best_model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def main(self, train_data):
        scoring = os.getenv("SCORING", "f1_macro")

        X_train = train_data.drop(self.target, axis=1)
        y_train = train_data[self.target]

        self.best_model = self._choose_best_model(X_train, y_train, scoring)
        self._train_best_model(X_train, y_train)
        self.save_model(self.best_model, self.model_path)
        preds = self.best_model.predict(X_train)
        return  pd.DataFrame(preds, index=X_train.index)


class ModelPredictionStep:

    def __init__(self, target, model_path='models/best_model.pickle'):
        self.target = target
        self.model_path = model_path
        
    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def main(self, train_data):
        if self.target in train_data.columns:
            train_data.drop(self.target, inplace=True)

        model = self.load_model(self.model_path)
        preds = model.predict(train_data)
        return pd.DataFrame(preds, index=train_data.index)
