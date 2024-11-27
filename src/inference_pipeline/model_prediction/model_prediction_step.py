import logging


logger = logging.getLogger(__name__)

class ModelPredictionStep:

    def __init__(self):
        pass

    def main(self, model, data_x):
        return model.predict(data_x)
