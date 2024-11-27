from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd


class PipelineStep(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def main(self, X_train: pd.DataFrame, 
             y_train: pd.DataFrame) -> Tuple[Optional[pd.DataFrame | pd.Series], Optional[pd.DataFrame| pd.Series]]:
        return X_train, y_train