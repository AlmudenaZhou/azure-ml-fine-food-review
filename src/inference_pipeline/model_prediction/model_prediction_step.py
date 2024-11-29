import logging
import pickle

import pandas as pd


logger = logging.getLogger(__name__)

class ModelPredictionStep:

    def __init__(self):
        pass

    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def main(self, model_path, data_x):
        model = self.load_model(model_path)
        predictions = model.predict(data_x)
        return pd.DataFrame(predictions, index=data_x.index)
