import logging
import pickle

import pandas as pd
import numpy as np

from word2vec_model import Word2Vec


logger = logging.getLogger(__name__)

class Text2VectorStep:

    def __init__(self):
        pass
    
    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        return model

    def main(self, model_path, data_x):
        model = self.load_model(model_path)
        data_transf = model.transform(data_x)
        if not isinstance(data_transf, np.ndarray):
            data_transf = data_transf.toarray()
        data_transf = pd.DataFrame(data_transf, index=data_x.index)
        return data_transf
