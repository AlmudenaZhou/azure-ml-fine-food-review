import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class ModelTrainingStep:

    def __init__(self):
        self.model_dict = {'logreg': LogisticRegression(),
                           'SVC': SVC(),
                           'dectree': DecisionTreeClassifier()}
        self.best_model = None
        
    def _choose_best_model(self, X_train, y_train,
                           scoring='f1_macro', cv=10):

        all_scores = [cross_val_score(model, X_train, y_train,
                                      cv=cv, scoring=scoring)
                    for model in self.model_dict.values()]
        all_results = pd.DataFrame(all_scores, index=self.model_dict.keys())
        best_model_name = all_results.mean(axis=1).idxmax()
        best_model = self.model_dict[best_model_name]
        return best_model
    
    def _train_best_model(self, X_train):
        self.best_model.fit(X_train)

    def main(self, X_train, y_train):
        self.best_model = self._choose_best_model(X_train, y_train)
        self._train_best_model(X_train)
        return self.best_model
