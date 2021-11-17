import numpy as np
import pandas as pd
from dataclasses import dataclass

from typing import Type, Union, List


@dataclass
class ModelData:
    model: Type
    data: Union[pd.DataFrame, np.ndarray]
    data_name: str
    excluded_columns: Union[List, None]
    test_data: Union[pd.DataFrame, np.ndarray] = None


@dataclass
class ModelPerformance:
    algorithm_name: str
    data_name: str
    downsampling_factor: float
    parameter: dict
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float

    def __str__(self):
        return f"Model: {self.algorithm_name}\nData: {self.data_name}\nDownsampling factor: {self.downsampling_factor}\n" \
               f"Parameter: {self.parameter}\nROC AUC: {self.roc_auc_mean:.3f}\u00B1{self.roc_auc_std:.3f}\nPR AUC:" \
               f"{self.pr_auc_mean:.3f}\u00B1{self.pr_auc_std:.3f}"
