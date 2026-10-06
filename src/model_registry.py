"""Run comparison, best-model selection, Model Registry and prediction.

Uses the current (MLflow 2.9+/3.x) registry workflow: *aliases* such as
"champion" instead of the deprecated Staging/Production stages.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

import config

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


class NoRunsFoundError(RuntimeError):
    """Raised when an experiment has no finished runs to choose from."""


# --------------------------------------------------------------------------
# 1. Compare runs / find the best one
# --------------------------------------------------------------------------
def compare_runs(experiment_name: str = config.EXPERIMENT_NAME,
                 primary: str = config.PRIMARY_METRIC,
                 secondary: str = config.SECONDARY_METRIC) -> pd.DataFrame:
    """Return all FINISHED runs as a table, best first (F1, then ROC-AUC)."""
    if mlflow.get_experiment_by_name(experiment_name) is None:
        raise NoRunsFoundError(f"Experiment '{experiment_name}' does not exist. "
                               "Run `python main.py` first.")
    # search_runs is the programmatic version of the MLflow UI table.
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{primary} DESC", f"metrics.{secondary} DESC"],
    )
    if runs.empty or f"metrics.{primary}" not in runs.columns:
        raise NoRunsFoundError(f"No finished runs with '{primary}' in '{experiment_name}'")
    runs = runs.dropna(subset=[f"metrics.{primary}"])
    if runs.empty:
        raise NoRunsFoundError(f"No runs have the metric '{primary}'")

    table = pd.DataFrame({
        "run_id": runs["run_id"],
        "model_name": runs["params.model_name"],
        **{m: runs[f"metrics.{m}"] for m in METRIC_COLUMNS},
        "training_time_seconds": runs.get("metrics.training_time_seconds"),
        "logged_model_uri": runs.get("tags.logged_model_uri"),
    })
    # Explicit sort too, so the ordering never depends on the backend's NULL handling.
    table = table.sort_values([primary, secondary], ascending=False).reset_index(drop=True)
    return table


def find_best_run(experiment_name: str = config.EXPERIMENT_NAME) -> dict:
    """Best run = highest F1; ties are broken by higher ROC-AUC."""
    best = compare_runs(experiment_name).iloc[0].to_dict()
    best["model_uri"] = best.get("logged_model_uri") or f"runs:/{best['run_id']}/model"
    return best


def summarise_winners(table: pd.DataFrame) -> dict[str, str]:
    """Answer 'which model is best on each metric?'."""
    return {m: table.loc[table[m].idxmax(), "model_name"]
            for m in ("accuracy", "f1_score", "roc_auc")}


def save_comparison(table: pd.DataFrame, out_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Write a CSV and a bar chart of the comparison for the documentation."""
    out_dir = Path(out_dir or config.ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "run_comparison.csv"
    table.to_csv(csv_path, index=False)

    # Latest run per model so repeated executions do not clutter the chart.
    latest = table.drop_duplicates("model_name")
    x = np.arange(len(latest))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(["accuracy", "f1_score", "roc_auc"]):
        ax.bar(x + (i - 1) * width, latest[m], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(latest["model_name"])
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Model comparison (test set)")
    ax.legend()
    fig.tight_layout()
    png_path = out_dir / "run_comparison.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return csv_path, png_path


# --------------------------------------------------------------------------
# 2. Register
# --------------------------------------------------------------------------
def register_best_model(best: dict,
                        registered_name: str = config.REGISTERED_MODEL_NAME,
                        alias: str = config.MODEL_ALIAS) -> dict:
    """Register the best run's model and point `alias` at the new version."""
    client = MlflowClient()

    # Re-running the pipeline should not create a duplicate version for the same run.
    existing = [mv for mv in client.search_model_versions(f"name='{registered_name}'")
                if mv.run_id == best["run_id"]]
    if existing:
        mv = existing[0]
        logger.info("Run %s already registered as version %s", best["run_id"], mv.version)
    else:
        # Creates the registered model on first use and a NEW VERSION every time.
        mv = mlflow.register_model(model_uri=best["model_uri"], name=registered_name)
        logger.info("Registered %s version %s", registered_name, mv.version)

    client.update_model_version(
        name=registered_name, version=mv.version,
        description=(f"{best['model_name']} selected by {config.PRIMARY_METRIC}="
                     f"{best[config.PRIMARY_METRIC]:.4f}, "
                     f"{config.SECONDARY_METRIC}={best[config.SECONDARY_METRIC]:.4f}"))
    client.set_model_version_tag(registered_name, mv.version, "model_name", best["model_name"])
    # Alias = a movable pointer. 'models:/<name>@champion' always resolves to it.
    client.set_registered_model_alias(registered_name, alias, mv.version)
    return {"name": registered_name, "version": int(mv.version), "alias": alias,
            "run_id": best["run_id"], "model_name": best["model_name"]}


# --------------------------------------------------------------------------
# 3. Load + predict
# --------------------------------------------------------------------------
def registered_model_uri(name: str, alias: str | None = None, version: int | None = None) -> str:
    if (alias is None) == (version is None):
        raise ValueError("Pass exactly one of alias or version")
    return f"models:/{name}@{alias}" if alias else f"models:/{name}/{version}"


def load_registered_model(name: str = config.REGISTERED_MODEL_NAME,
                          alias: str | None = config.MODEL_ALIAS,
                          version: int | None = None):
    """Load the model FROM THE REGISTRY (pyfunc = framework-independent wrapper)."""
    if version is not None:
        alias = None
    uri = registered_model_uri(name, alias, version)
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception as exc:  # surface a readable message to beginners
        raise RuntimeError(f"Could not load '{uri}'. Was the model registered? ({exc})") from exc


def predict_with_registered_model(model, X: pd.DataFrame) -> np.ndarray:
    if X is None or len(X) == 0:
        raise ValueError("No rows supplied for prediction")
    return np.asarray(model.predict(X))


def export_best_model_copy(name: str = config.REGISTERED_MODEL_NAME,
                           alias: str = config.MODEL_ALIAS,
                           out_dir: Path | str | None = None) -> Path:
    """Save a plain joblib copy of the registered model into models/."""
    out_dir = Path(out_dir or config.MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sk_model = mlflow.sklearn.load_model(registered_model_uri(name, alias))
    path = out_dir / "best_model.joblib"
    joblib.dump(sk_model, path)
    return path


def save_registry_info(info: dict, out_dir: Path | str | None = None) -> Path:
    out_dir = Path(out_dir or config.MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "registry_info.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path
