import logging


logger = logging.getLogger(__name__)

class Text2VectorStep:

    def __init__(self):
        pass

    def main(self, model, data_x):
        return model.transform(data_x)
