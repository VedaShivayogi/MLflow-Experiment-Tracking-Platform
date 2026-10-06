"""MLflow tracking helpers: experiment setup and the generic 'one run' routine.

Every train_*.py file calls `run_experiment`, so the logging logic exists once.
"""
from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.models import infer_signature

import config
from src import evaluation
from src.data_preprocessing import DataSplit

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    model_name: str
    metrics: dict
    model_uri: str


def setup_mlflow(tracking_uri: str | None = None,
                 artifact_root: Path | str | None = None,
                 experiment_name: str = config.EXPERIMENT_NAME) -> str:
    """Point MLflow at the SQLite DB and create/select the experiment.

    Returns the experiment id. Safe to call repeatedly.
    """
    tracking_uri = tracking_uri or config.TRACKING_URI
    artifact_root = Path(artifact_root or config.ARTIFACT_ROOT)
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)  # where runs/params/metrics are stored
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        # artifact_location is fixed at creation time; as_uri() makes a valid
        # file:///C:/... URI on Windows.
        exp_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_root.resolve().as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking_uri=%s experiment=%s", tracking_uri, experiment_name)
    return exp_id


# scikit-learn tree models contain this type. MLflow 3.x saves sklearn models with
# the "skops" format by default and refuses to load types it has not been told to
# trust. We created these files ourselves, so trusting this one type is safe.
TRUSTED_SKOPS_TYPES = ["sklearn.tree._tree.Tree"]


def _log_model(model, signature, input_example):
    """Log an sklearn model, coping with differences between MLflow versions.

    * MLflow 3.x : `name=` and `skops_trusted_types=`
    * MLflow 2.x : `artifact_path=` (pickle format, no trusted types needed)
    """
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {"signature": signature, "input_example": input_example}
    kwargs["name" if "name" in params else "artifact_path"] = "model"
    if "skops_trusted_types" in params:
        kwargs["skops_trusted_types"] = TRUSTED_SKOPS_TYPES
    return mlflow.sklearn.log_model(model, **kwargs)


def run_experiment(model_name: str, estimator, hyperparams: dict, data: DataSplit,
                   model_family: str, run_name: str | None = None,
                   local_artifact_dir: Path | str | None = None) -> RunResult:
    """Train `estimator`, evaluate it and log everything to ONE MLflow run."""
    local_dir = Path(local_artifact_dir or config.ARTIFACTS_DIR / "runs") / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # `with` guarantees the run is closed (and marked FAILED on exceptions).
    with mlflow.start_run(run_name=run_name or model_name) as run:
        # ---- Tags: free-text labels, handy for filtering in the UI ----------
        mlflow.set_tags({"model_family": model_family, "dataset": config.DATASET_NAME,
                         "task": "binary_classification"})

        # ---- Parameters: the inputs that define this run --------------------
        mlflow.log_params({
            "model_name": model_name,
            "test_size": data.test_size,
            "random_state": data.random_state,
            "n_train_samples": len(data.X_train),
            "n_test_samples": len(data.X_test),
            "n_features": data.X_train.shape[1],
            **hyperparams,
        })

        # ---- Train -----------------------------------------------------------
        start = time.perf_counter()
        estimator.fit(data.X_train, data.y_train)
        train_seconds = time.perf_counter() - start

        # ---- Evaluate --------------------------------------------------------
        y_pred = estimator.predict(data.X_test)
        y_proba = estimator.predict_proba(data.X_test)[:, 1]
        metrics = evaluation.compute_metrics(data.y_test, y_pred, y_proba)

        # ---- Metrics: the results that let us compare runs ------------------
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", train_seconds)

        # ---- Artifacts: files attached to the run ----------------------------
        files = [
            evaluation.plot_confusion_matrix(
                data.y_test, y_pred, local_dir / "confusion_matrix.png",
                f"{model_name}: confusion matrix"),
            evaluation.plot_roc_curve(
                data.y_test, y_proba, local_dir / "roc_curve.png",
                f"{model_name}: ROC curve"),
            evaluation.save_classification_report(
                data.y_test, y_pred, local_dir / "classification_report.txt"),
            evaluation.plot_feature_importance(
                estimator, data.feature_names, local_dir / "feature_importance.png",
                f"{model_name}: top features"),
        ]
        for f in files:
            if f is not None:  # feature importance may not exist for some models
                mlflow.log_artifact(str(f), artifact_path="evaluation")

        # ---- Model logging: saves the trained model + its input/output schema --
        signature = infer_signature(data.X_train, estimator.predict(data.X_train))
        info = _log_model(estimator, signature, data.X_train.head(3))
        mlflow.set_tag("logged_model_uri", info.model_uri)

        logger.info("%s -> run_id=%s f1=%.4f roc_auc=%.4f", model_name,
                    run.info.run_id, metrics["f1_score"], metrics["roc_auc"])
        return RunResult(run.info.run_id, model_name, metrics, info.model_uri)
