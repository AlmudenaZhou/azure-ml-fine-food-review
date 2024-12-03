import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

class SplitDataStep:

    def __init__(self, target: str):
        self.target = target

    def main(self, data, test_size=float(os.getenv("TEST_SIZE", "0.15"))):
        logger.info("Splitting the data...")

        X_data = data.drop(self.target, axis=1)
        y_data = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        return train_data, test_data
