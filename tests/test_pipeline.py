"""Pipeline tests. They use a throw-away SQLite DB in a temp folder, so they never
touch your real mlflow.db. Small hyperparameters keep the suite fast."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src import (data_preprocessing, model_registry, train_gradient_boosting,
                 train_logistic, train_random_forest)
from src.tracking import setup_mlflow

EXP = "pytest_experiment"
REG_NAME = "Pytest_Best_Model"
METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


@pytest.fixture(scope="module")
def data():
    return data_preprocessing.get_data()


@pytest.fixture(scope="module")
def env(tmp_path_factory, data):
    """Isolated MLflow backend + the three runs, executed once for the module."""
    base = tmp_path_factory.mktemp("mlflow_env")
    uri = "sqlite:///" + (base / "mlflow.db").as_posix()
    setup_mlflow(uri, base / "artifacts", EXP)
    kw = {"local_artifact_dir": base / "plots"}
    results = [
        train_logistic.train(data, **kw),
        train_random_forest.train(data, {"n_estimators": 30}, **kw),
        train_gradient_boosting.train(data, {"n_estimators": 30}, **kw),
    ]
    return {"base": base, "uri": uri, "results": results}


# ---- data ---------------------------------------------------------------
def test_dataset_loads():
    df = data_preprocessing.load_dataset()
    assert df.shape == (569, 31)
    assert not df.isnull().any().any()


def test_split_is_stratified_and_reproducible(data):
    again = data_preprocessing.get_data()
    assert len(data.X_train) + len(data.X_test) == 569
    pd.testing.assert_frame_equal(data.X_train, again.X_train)
    assert abs(data.y_train.mean() - data.y_test.mean()) < 0.02


def test_invalid_test_size_raises():
    with pytest.raises(ValueError):
        data_preprocessing.split_data(data_preprocessing.load_dataset(), test_size=1.5)


# ---- training / metrics / runs / artifacts --------------------------------
def test_three_models_trained_with_valid_metrics(env):
    assert len(env["results"]) == 3
    for r in env["results"]:
        assert set(METRICS) <= set(r.metrics)
        assert all(0.0 <= r.metrics[m] <= 1.0 for m in METRICS)
        assert r.metrics["accuracy"] > 0.9  # sanity: this dataset is easy


def test_mlflow_runs_created_with_params_metrics(env):
    mlflow.set_tracking_uri(env["uri"])
    runs = mlflow.search_runs(experiment_names=[EXP])
    assert len(runs) == 3
    assert set(runs["status"]) == {"FINISHED"}
    assert set(runs["params.model_name"]) == {
        "LogisticRegression", "RandomForest", "GradientBoosting"}
    for col in ["params.test_size", "params.random_state"] + [f"metrics.{m}" for m in METRICS]:
        assert runs[col].notna().all(), col


def test_artifacts_logged_for_every_run(env):
    mlflow.set_tracking_uri(env["uri"])
    client = MlflowClient()
    expected = {"confusion_matrix.png", "roc_curve.png",
                "classification_report.txt", "feature_importance.png"}
    for r in env["results"]:
        names = {Path(a.path).name for a in client.list_artifacts(r.run_id, "evaluation")}
        assert expected <= names


def test_model_artifact_logged_for_every_run(env):
    """MLflow 3 stores models as 'logged models' linked to the run."""
    mlflow.set_tracking_uri(env["uri"])
    for r in env["results"]:
        loaded = mlflow.sklearn.load_model(r.model_uri)
        assert hasattr(loaded, "predict")


# ---- best model / registry / prediction -----------------------------------
def test_best_model_is_highest_f1(env):
    mlflow.set_tracking_uri(env["uri"])
    best = model_registry.find_best_run(EXP)
    top_f1 = max(r.metrics["f1_score"] for r in env["results"])
    assert best["f1_score"] == pytest.approx(top_f1)
    table = model_registry.compare_runs(EXP)
    assert table["f1_score"].is_monotonic_decreasing


def test_no_runs_raises_clear_error(tmp_path):
    uri = "sqlite:///" + (tmp_path / "empty.db").as_posix()
    mlflow.set_tracking_uri(uri)
    with pytest.raises(model_registry.NoRunsFoundError):
        model_registry.compare_runs("does_not_exist")


def test_register_load_and_predict(env, data):
    mlflow.set_tracking_uri(env["uri"])
    best = model_registry.find_best_run(EXP)
    info = model_registry.register_best_model(best, REG_NAME, "champion")
    assert info["version"] == 1

    # Registering the same run again must not create a duplicate version.
    assert model_registry.register_best_model(best, REG_NAME, "champion")["version"] == 1

    model = model_registry.load_registered_model(REG_NAME, alias="champion")
    sample = data.X_test.head(20)
    preds = model_registry.predict_with_registered_model(model, sample)
    assert preds.shape == (20,)
    assert set(np.unique(preds)) <= {0, 1}
    assert (preds == data.y_test.head(20).values).mean() >= 0.8

    # Same model reachable by explicit version number.
    by_version = model_registry.load_registered_model(REG_NAME, alias=None, version=1)
    assert np.array_equal(by_version.predict(sample), preds)


def test_load_unregistered_model_raises(env):
    mlflow.set_tracking_uri(env["uri"])
    with pytest.raises(RuntimeError):
        model_registry.load_registered_model("Nope_Model", alias="champion")


def test_empty_prediction_input_rejected(env):
    mlflow.set_tracking_uri(env["uri"])
    model = model_registry.load_registered_model(REG_NAME, alias="champion")
    with pytest.raises(ValueError):
        model_registry.predict_with_registered_model(model, pd.DataFrame())
