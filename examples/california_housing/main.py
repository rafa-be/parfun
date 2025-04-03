"""
This example trains a random tree regressor on the California housing dataset from scikit-learn.

Measure the training time when splitting the learning dataset process using Parfun.

    $ python -m examples.california_housing.main
"""

import psutil
import timeit

from typing import List

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor

from parfun.decorators import parfun
from parfun.entry_point import set_parallel_backend_context
from parfun.partition.api import per_argument
from parfun.partition.dataframe import df_by_row


class MeanRegressor(RegressorMixin):
    def __init__(self, regressors: List[RegressorMixin]) -> None:
        super().__init__()
        self._regressors = regressors

    def predict(self, X):
        return np.mean([regressor.predict(X) for regressor in self._regressors])


@parfun(split=per_argument(dataframe=df_by_row), combine_with=lambda regressors: MeanRegressor(list(regressors)))
def train_regressor(dataframe: pd.DataFrame, feature_names: List[str], target_name: str) -> RegressorMixin:

    regressor = DecisionTreeRegressor()
    regressor.fit(dataframe[feature_names], dataframe[[target_name]])

    return regressor


if __name__ == "__main__":
    N_WORKERS = psutil.cpu_count(logical=False)

    dataset = fetch_california_housing(download_if_missing=True)

    feature_names = dataset["feature_names"]
    target_name = dataset["target_names"][0]

    dataframe = pd.DataFrame(dataset["data"], columns=feature_names)
    dataframe[target_name] = dataset["target"]

    N_MEASURES = 5

    with set_parallel_backend_context("local_single_process"):
        regressor = train_regressor(dataframe, feature_names, target_name)

        duration = (
            timeit.timeit(lambda: train_regressor(dataframe, feature_names, target_name), number=N_MEASURES)
            / N_MEASURES
        )

        print("Sequential training duration:", duration)

    with set_parallel_backend_context("local_multiprocessing", max_workers=N_WORKERS):
        regressor = train_regressor(dataframe, feature_names, target_name)

        duration = (
            timeit.timeit(lambda: train_regressor(dataframe, feature_names, target_name), number=N_MEASURES)
            / N_MEASURES
        )

        print("Parallel training duration:", duration)
