"""Experiment 1 - Logistic Regression (the baseline model)."""
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data_preprocessing import DataSplit, get_data
from src.tracking import RunResult, run_experiment, setup_mlflow

MODEL_NAME = "LogisticRegression"


def build_model(params: dict) -> Pipeline:
    # Logistic regression needs scaled inputs, so scaling is part of the model
    # pipeline (this also means the registered model scales new data itself).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=config.RANDOM_STATE, **params)),
    ])


def train(data: DataSplit, params: dict | None = None, **kwargs) -> RunResult:
    params = {**config.LOGISTIC_PARAMS, **(params or {})}
    return run_experiment(MODEL_NAME, build_model(params), params, data,
                          model_family="linear", **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_mlflow()
    print(train(get_data()))
