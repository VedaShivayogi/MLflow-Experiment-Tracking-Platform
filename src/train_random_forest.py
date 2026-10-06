"""Experiment 2 - Random Forest."""
import logging

from sklearn.ensemble import RandomForestClassifier

import config
from src.data_preprocessing import DataSplit, get_data
from src.tracking import RunResult, run_experiment, setup_mlflow

MODEL_NAME = "RandomForest"


def build_model(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1, **params)


def train(data: DataSplit, params: dict | None = None, **kwargs) -> RunResult:
    params = {**config.RANDOM_FOREST_PARAMS, **(params or {})}
    return run_experiment(MODEL_NAME, build_model(params), params, data,
                          model_family="tree_ensemble_bagging", **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_mlflow()
    print(train(get_data()))
