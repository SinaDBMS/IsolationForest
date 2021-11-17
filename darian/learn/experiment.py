import logging
import pickle
from abc import ABC, abstractmethod
import numpy as np
from typing import final, List, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import roc_auc_score
from darian.util.evaluation import precision_recall_auc_score
from darian.learn.data_structures import ModelData, ModelPerformance
from darian.learn.parameter import ParameterGrid


class Experiment(ABC):

    def __init__(self):
        self.__model_performance: List[ModelPerformance] = []
        self.__logger = logging.getLogger(f"{__name__}:{self.__class__.__name__}")

    @property
    def model_performance(self):
        return self.__model_performance

    @property
    def roc_performance_summary(self):
        summary = {}
        for model_performance in self.__model_performance:
            model_data_name = f"{model_performance.algorithm_name} - {model_performance.data_name}"
            key = (model_performance.parameter.__str__(), model_data_name)
            if key not in summary.keys():
                summary[key] = pd.Series(dtype='float')
                summary[key].index = summary[key].index.astype(int)
            summary[key][model_performance.downsampling_factor] = model_performance.roc_auc_mean
        return pd.DataFrame(summary)

    @property
    def pr_performance_summary(self):
        summary = {}
        for model_performance in self.__model_performance:
            model_data_name = f"{model_performance.algorithm_name} - {model_performance.data_name}"
            key = (model_performance.parameter.__str__(), model_data_name)
            if key not in summary.keys():
                summary[key] = pd.Series(dtype='float')
                summary[key].index = summary[key].index.astype(int)
            summary[key][model_performance.downsampling_factor] = model_performance.pr_auc_mean
        return pd.DataFrame(summary)

    @final
    def perform_experiment(self):
        for downsampling_factor in self.get_downsampling_factor():
            for model_data in self.generate_model_data():
                algorithm_name = model_data.model().__str__()
                self.__logger.info(f"Performing experiments on {model_data.data_name} using {algorithm_name}. "
                                   f"Downsampling factor: {downsampling_factor:.2f}")
                indices_to_be_removed = self.__downsample(downsampling_factor, 1)
                training_data = model_data.data.drop(indices_to_be_removed)

                for parameters in self.get_parameters():
                    self.__logger.info(f"Parameters: {parameters}")
                    roc_auc = []
                    pr_auc = []
                    i = 1
                    parameters_copy = parameters.copy()
                    for random_state in self.random_states():
                        self.__logger.info(f"RandomState {i} of {len(self.random_states())}")
                        parameters_copy['random_state'] = random_state
                        parameters_copy['n_jobs'] = self.get_n_jobs()
                        model = model_data.model()
                        model.set_params(**parameters_copy)
                        model.fit(training_data, excluded_columns=model_data.excluded_columns)
                        test_data = model_data.test_data if model_data.test_data is not None else model_data.data
                        scores = model.score_samples(test_data)
                        roc_auc.append(roc_auc_score(self.get_label_column(), scores))
                        pr_auc.append(precision_recall_auc_score(self.get_label_column(), scores))
                        i += 1
                    roc_auc = np.array(roc_auc)
                    self.__model_performance.append(
                        ModelPerformance(algorithm_name, model_data.data_name, downsampling_factor, parameters,
                                         np.mean(roc_auc), np.std(roc_auc), np.mean(pr_auc), np.std(pr_auc)))

    def plot_auc_performance(self):
        summary = self.roc_performance_summary
        plt.figure()
        ax: Axes = plt.axes()
        ax.set_xlabel("Downsampling factor")
        ax.set_ylabel("AUC")
        for col in summary.columns:
            ax.plot(summary.index, summary[col], label=col)
        ax.legend()
        plt.show()

    @abstractmethod
    def get_parameters(self) -> ParameterGrid:
        pass

    @abstractmethod
    def generate_model_data(self) -> List[ModelData]:
        pass

    @staticmethod
    def get_downsampling_factor() -> List[Union[float, None]]:
        return [i for i in np.arange(0.1, 0, -0.01)]

    @abstractmethod
    def get_label_column(self) -> Union[pd.Series, np.ndarray]:
        pass

    @abstractmethod
    def get_label_name(self) -> Union[str, int]:
        pass

    @staticmethod
    def random_states():
        return [i for i in range(10000, 20000, 1000)]

    @staticmethod
    def get_n_jobs():
        return 11

    def __downsample(self, downsampling_factor: float, positive_class, random_state: int = 0):
        if downsampling_factor is None or downsampling_factor < 0:
            return []

        positive_instances = self.get_label_column()[self.get_label_column() == positive_class]
        number_of_positive_instances_to_be_removed = \
            int((positive_instances.shape[0] - downsampling_factor * self.get_label_column().shape[0])
                / (1 - downsampling_factor))
        instances_to_be_removed = positive_instances.sample(n=number_of_positive_instances_to_be_removed,
                                                            random_state=random_state)
        return instances_to_be_removed.index

    def save_experiment(self, location: str):
        with open(location, "wb") as output:
            pickle.dump(self, output)


def load_experiment(location: str) -> Experiment:
    with open(location, "rb") as input_location:
        return pickle.load(input_location)
