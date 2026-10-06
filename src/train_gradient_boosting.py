"""Experiment 3 - Gradient Boosting (scikit-learn; no extra dependency like XGBoost)."""
import logging

from sklearn.ensemble import GradientBoostingClassifier

import config
from src.data_preprocessing import DataSplit, get_data
from src.tracking import RunResult, run_experiment, setup_mlflow

MODEL_NAME = "GradientBoosting"


def build_model(params: dict) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=config.RANDOM_STATE, **params)


def train(data: DataSplit, params: dict | None = None, **kwargs) -> RunResult:
    params = {**config.GRADIENT_BOOSTING_PARAMS, **(params or {})}
    return run_experiment(MODEL_NAME, build_model(params), params, data,
                          model_family="tree_ensemble_boosting", **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_mlflow()
    print(train(get_data()))
