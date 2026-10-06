"""Metrics and evaluation artifacts (plots + text report). No MLflow calls here."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: works on any Windows laptop, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)

CLASS_NAMES = ["malignant", "benign"]  # index 0 / 1


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """The five required metrics. Positive class = 1 (benign)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_proba, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.legend(loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_classification_report(y_true, y_pred, path: Path) -> Path:
    text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    path.write_text(text, encoding="utf-8")
    return path


def _final_estimator(model):
    """Return the last step if `model` is a Pipeline, else the model itself."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def get_feature_importances(model) -> tuple[np.ndarray, str] | None:
    """Return (importances, label) or None if the model exposes neither."""
    est = _final_estimator(model)
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_), "Feature importance"
    if hasattr(est, "coef_"):
        return np.abs(np.ravel(est.coef_)), "|Coefficient| (scaled features)"
    return None


def plot_feature_importance(model, feature_names, path: Path, title: str,
                            top_n: int = 15) -> Path | None:
    found = get_feature_importances(model)
    if found is None:
        return None
    values, xlabel = found
    order = np.argsort(values)[::-1][:top_n][::-1]  # top N, largest at the top
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order], values[order], color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
