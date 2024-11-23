import os
import logging

from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

class SplitDataStep:

    def main(self, data, test_size=float(os.environ["TEST_SIZE"])):
        logger.info("Splitting the data...")
        test_size = test_size
        X_train, X_test, y_train, y_test = train_test_split(data.Text, data.Label, test_size=test_size)

        return X_train, X_test, y_train, y_test
