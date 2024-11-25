import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class ModelTrainingStep:

    def __init__(self, model_path='models/best_model.pickle'):
        self.model_dict = {'logreg': LogisticRegression(),
                           'SVC': SVC(),
                           'dectree': DecisionTreeClassifier()}
        self.best_model = None
        self.model_path = model_path
        
    def _choose_best_model(self, X_train, y_train,
                           scoring='f1_macro', cv=10):

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

    def main(self, X_train, y_train):
        scoring = os.environ["SCORING"]
        self.best_model = self._choose_best_model(X_train, y_train, scoring)
        self._train_best_model(X_train, y_train)
        self.save_model(self.best_model, self.model_path)
        return self.best_model
