import logging
from multiprocessing import Pool
from numbers import Number
from typing import List, Tuple, Dict, Union, Any
from dataclasses import dataclass
from random import Random
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype, is_bool_dtype
import math
from sklearn.feature_extraction.text import CountVectorizer
from darian.util import configuration
from darian.util.text_preprocessing import extract_n_grams

configuration.configure_logger()


class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, contamination: float = 0.05,
                 bootstrap: bool = False, string_features_parameters: Dict[Union[str, Tuple], Dict[str, Any]] = None,
                 categorical_feature_subset_length: Union[str, int] = 'random', jaccard_similarity_threshold: float = 0,
                 random_state: int = 0, n_jobs: int = 1):
        """
        Isolation forest algorithm.

        :param n_estimators: The number of base estimators in the ensemble.
        :param max_samples: The number of samples to draw from input_data to train each base estimator.
        :param contamination: The proportion of anomalies in the training data set.
        :param bootstrap Whether to sample the subsample data set with replacement or not.
        :param string_features_parameters: A dictionary, which specifies the parameters for text processing for each
        string feature in the data set.
        :param categorical_feature_subset_length:
        :param jaccard_similarity_threshold: Two sets are similar if their jaccard similarity is greater than this value
        :param random_state:
        :param n_jobs: Number of processes to run in parallel when scoring.
        """
        self.__n_estimators = n_estimators
        self.__max_samples = max_samples
        self.__contamination = contamination
        self.__bootstrap = bootstrap
        self.__random_state = random_state
        self.__n_jobs = n_jobs
        self.__string_features_parameters: Dict[Union[str, Tuple], Dict[str, Any]] = string_features_parameters
        self.__categorical_feature_subset_length: Union[str, int] = categorical_feature_subset_length
        self.__jaccard_similarity_threshold = jaccard_similarity_threshold
        self.forest: List[IsolationTree] = []
        self.__logger = logging.getLogger(f"{__name__}:{self.__class__.__name__}")

    def fit(self, X: pd.DataFrame, excluded_columns: List = None):
        x_without_excluded_columns: pd.DataFrame = X.drop([] if excluded_columns is None else excluded_columns, axis=1)
        height_limit = math.ceil(math.log2(self.__max_samples))
        forest = []
        for i in range(self.__n_estimators):
            subsampled_x = x_without_excluded_columns.sample(n=self.__max_samples, replace=self.__bootstrap,
                                                             random_state=self.__random_state)
            isolation_tree = IsolationTree(X=subsampled_x, current_tree_height=0, height_limit=height_limit,
                                           string_features_parameters=self.__string_features_parameters,
                                           categorical_feature_subset_length=self.__categorical_feature_subset_length,
                                           jaccard_similarity_threshold=self.__jaccard_similarity_threshold,
                                           random_state=Random(self.__random_state + i))
            forest.append(isolation_tree)
        with Pool(self.__n_jobs) as p:
            self.forest = p.map(_fit, [iTree for iTree in forest])
        pass

    def score_samples(self, X: pd.DataFrame):
        path_length = pd.DataFrame()
        with Pool(self.__n_jobs) as p:
            results = p.starmap(_score_samples, [(iTree, X) for iTree in self.forest])
        for result in results:
            path_length = path_length.append(result, ignore_index=True)
        avg_path_lengths = path_length.mean()
        anomaly_scores = pd.DataFrame(
            np.power(2, -avg_path_lengths / _average_path_length(self.__max_samples)))
        return anomaly_scores

    def set_params(self, **params):
        for key in params.keys():
            setattr(self, f"_{self.__class__.__name__}__{key}", params[key])

    def __str__(self):
        return "Isolation Forest"


class IsolationTree:
    def __init__(self, X: pd.DataFrame, current_tree_height: int, height_limit: int,
                 string_features_parameters: Dict[Union[str, Tuple], Dict[str, Any]] = None,
                 categorical_feature_subset_length: Union[str, int] = 'random', jaccard_similarity_threshold: float = 0,
                 random_state: Random = Random(0)):
        """
        Builds a single Isolation Tree on the given data set.

        :param X: The training dataset.
        :param current_tree_height: the height of the current tree.
        :param height_limit: Maximum height allowed.
        :param string_features_parameters: A dictionary, which specifies the parameters for text processing for each
        string feature in the data set.
        :param categorical_feature_subset_length
        :param jaccard_similarity_threshold: Two sets are similar if their jaccard similarity is greater than this value
        :param random_state:
        """
        self.X = X
        self.current_tree_height = current_tree_height
        self.height_limit = height_limit
        self.left_sub_tree: IsolationTree = None
        self.right_sub_tree: IsolationTree = None
        self.split_attribute = None
        self.split_value = None
        self.marginal_distance = None
        self.random_point = None
        self.string_features_parameters: Dict[Union[str, Tuple], Dict[str, Any]] = string_features_parameters
        self.categorical_feature_subset_length: Union[str, int] = categorical_feature_subset_length
        self.jaccard_similarity_threshold = jaccard_similarity_threshold
        self.is_string_feature: bool = False
        self.leaf_node: LeafNode = None
        self.__random_state = random_state
        self.__logger = logging.getLogger(f"{__name__}:{self.__class__.__name__}")

    def build_isolation_tree(self):
        if self.current_tree_height >= self.height_limit or self.X.shape[0] <= 1:
            self.leaf_node = LeafNode(size=self.X.shape[0],
                                      path_length=self.current_tree_height + _average_path_length(self.X.shape[0]))
        else:
            self.split_attribute = self.__select_random_feature()
            random_column: pd.Series = self.X.loc[:, self.split_attribute].squeeze()

            left_child_node_elements: pd.DataFrame = None
            right_child_node_elements: pd.DataFrame = None
            if is_categorical_dtype(random_column):
                possible_values = random_column.value_counts(dropna=False)
                possible_non_zero_values = possible_values[possible_values > 0].index.tolist()
                if self.categorical_feature_subset_length == 'random':
                    random_subset_size = self.__random_state.randint(1, len(possible_non_zero_values))
                else:
                    random_subset_size = self.categorical_feature_subset_length

                self.split_value = self.__random_state.sample(possible_non_zero_values, random_subset_size)
                mask = random_column.isin(self.split_value)
                left_child_node_elements = self.X[mask]
                right_child_node_elements = self.X[~mask]
            elif is_bool_dtype(random_column):
                left_child_node_elements = self.X[random_column]
                right_child_node_elements = self.X[~random_column]
            elif is_numeric_dtype(random_column):
                min_value = float(np.min(random_column))
                max_value = float(np.max(random_column))
                self.split_value = self.__random_state.random() * (max_value - min_value) + min_value

                mask = random_column <= self.split_value
                left_child_node_elements = self.X[mask]
                right_child_node_elements = self.X[~mask]
            elif is_string_dtype(random_column):
                self.is_string_feature = True
                parameters = self.string_features_parameters[self.split_attribute].copy()
                splitter = parameters.pop('splitter')
                vectorizer = CountVectorizer(token_pattern=rf"(?u)[{splitter}]([^{splitter}]+)[{splitter}]",
                                             **parameters)
                try:
                    vectorizer.fit(random_column)
                    features = vectorizer.get_feature_names()
                    random_subset_size = self.__random_state.randint(1, len(features))
                    self.split_value = self.__random_state.sample(features, random_subset_size)
                except ValueError:
                    self.split_value = []

                mask = random_column.apply(
                    lambda row: _coverage_of_intersection(
                        extract_n_grams(row, splitter, vectorizer.lowercase, vectorizer.ngram_range),
                        self.split_value) > self.jaccard_similarity_threshold)
                left_child_node_elements = self.X[mask]
                right_child_node_elements = self.X[~mask]
            else:
                self.__logger.error(f"Unknown data type. Column: {self.split_attribute}")

            left_sub_tree = IsolationTree(left_child_node_elements, self.current_tree_height + 1, self.height_limit,
                                          self.string_features_parameters, self.categorical_feature_subset_length,
                                          self.jaccard_similarity_threshold, self.__random_state)
            right_sub_tree = IsolationTree(right_child_node_elements, self.current_tree_height + 1, self.height_limit,
                                           self.string_features_parameters, self.categorical_feature_subset_length,
                                           self.jaccard_similarity_threshold, self.__random_state)
            left_sub_tree.build_isolation_tree()
            right_sub_tree.build_isolation_tree()
            self.left_sub_tree = left_sub_tree
            self.right_sub_tree = right_sub_tree

    def __select_random_feature(self) -> Union[Tuple, str]:
        df = self.X
        col_path: List = []
        while isinstance(df, pd.DataFrame) and df.columns.nlevels > 0:
            current_level_column_values = list(df.columns.get_level_values(0).value_counts().index)
            random_col = self.__random_state.choice(current_level_column_values)
            col_path.append(random_col)
            df = df[random_col]

        index = tuple(col_path) if len(col_path) > 1 else col_path[0]
        return index

    def score_samples(self, X: pd.DataFrame):
        return X.apply(self.__get_path_of_single_instance, axis=1)

    def __get_path_of_single_instance(self, X: pd.Series) -> float:
        if self.leaf_node is not None:
            return self.leaf_node.path_length

        if isinstance(self.split_value, list):
            if self.is_string_feature:
                lowercase = True
                ngram_range = (1, 1)
                splitter = self.string_features_parameters[self.split_attribute]['splitter']
                if 'lowercase' in self.string_features_parameters[self.split_attribute].keys():
                    lowercase = self.string_features_parameters[self.split_attribute]['lowercase']
                if 'ngram_range' in self.string_features_parameters[self.split_attribute].keys():
                    ngram_range = self.string_features_parameters[self.split_attribute]['ngram_range']

                n_grams = extract_n_grams(X[self.split_attribute], splitter, lowercase, ngram_range)
                if _coverage_of_intersection(n_grams, self.split_value) > self.jaccard_similarity_threshold:
                    return self.left_sub_tree.__get_path_of_single_instance(X)
                else:
                    return self.right_sub_tree.__get_path_of_single_instance(X)
            else:
                if X[self.split_attribute] in self.split_value:
                    return self.left_sub_tree.__get_path_of_single_instance(X)
                else:
                    return self.right_sub_tree.__get_path_of_single_instance(X)
        elif isinstance(self.split_value, bool):
            if X[self.split_attribute]:
                return self.left_sub_tree.__get_path_of_single_instance(X)
            else:
                return self.right_sub_tree.__get_path_of_single_instance(X)
        elif isinstance(self.split_value, Number):
            if X[self.split_attribute] <= self.split_value:
                return self.left_sub_tree.__get_path_of_single_instance(X)
            else:
                return self.right_sub_tree.__get_path_of_single_instance(X)


def _fit(iTree: IsolationTree):
    iTree.build_isolation_tree()
    return iTree


def _score_samples(iTree: IsolationTree, X: pd.DataFrame):
    return iTree.score_samples(X)


@dataclass
class LeafNode:
    size: int
    path_length: float


def _average_path_length(n: int):
    if n <= 1:
        return 0
    elif n == 2:
        return 1
    else:
        def harmonic_number(i):
            return np.log(i) + np.euler_gamma

        return 2 * harmonic_number(n - 1) - 2 * (n - 1) / n


def _coverage_of_intersection(set1, set2):
    """
    Coverage of intersection of two sets is calculated as follows: len(set1 & set2) / len(set2)
    In other words, what percentage of the intersection of set1 and set2 is to be found in set2.
    :param set1
    :param set2
    """
    set1 = set(set1)
    set2 = set(set2)
    if '' in set1:
        set1.remove('')
    if '' in set2:
        set2.remove('')

    if len(set2) == 0:
        return 0
    intersection_size = len(set1 & set2)
    return intersection_size / len(set2)
