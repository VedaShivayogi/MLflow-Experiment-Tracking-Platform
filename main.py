"""End-to-end pipeline:
data -> 3 MLflow runs -> compare -> best model -> registry -> load -> predict.

    python main.py                 # everything
    python main.py --registry-only # skip training, reuse existing runs
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

# Harmless pydantic warning raised inside MLflow itself on some installs.
warnings.filterwarnings("ignore", message=".*protected namespace.*")

import pandas as pd

import config
from src import (data_preprocessing, model_registry, train_gradient_boosting,
                 train_logistic, train_random_forest)
from src.tracking import setup_mlflow

log = logging.getLogger("main")


def run_training(data) -> list:
    """Experiments 1-3: one separate MLflow run per model."""
    results = []
    for trainer in (train_logistic, train_random_forest, train_gradient_boosting):
        results.append(trainer.train(data))
    return results


def run_registry_demo(data) -> dict:
    """Compare -> select -> register -> load -> predict."""
    table = model_registry.compare_runs()
    pd.set_option("display.width", 200)
    print("\n=== Run comparison (best first) ===")
    print(table.drop(columns=["run_id", "logged_model_uri"]).round(4).to_string(index=False))
    winners = model_registry.summarise_winners(table)
    print("\nHighest accuracy :", winners["accuracy"])
    print("Highest F1-score :", winners["f1_score"])
    print("Highest ROC-AUC  :", winners["roc_auc"])
    csv_path, png_path = model_registry.save_comparison(table)
    print(f"Saved comparison: {csv_path} and {png_path}")

    best = model_registry.find_best_run()
    print(f"\nBEST MODEL: {best['model_name']} (run {best['run_id']}) "
          f"F1={best['f1_score']:.4f} ROC-AUC={best['roc_auc']:.4f}")

    info = model_registry.register_best_model(best)
    print(f"Registered '{info['name']}' version {info['version']} "
          f"(alias '{info['alias']}')")

    # ---- Load the model from the registry and predict -----------------------
    model = model_registry.load_registered_model(info["name"], alias=info["alias"])
    sample = data.X_test.head(5)
    preds = model_registry.predict_with_registered_model(model, sample)
    print("\n=== Prediction with registered model ===")
    out = pd.DataFrame({"predicted": preds, "actual": data.y_test.head(5).values})
    out["label"] = out["predicted"].map({0: "malignant", 1: "benign"})
    print(out.to_string(index=False))

    info["f1_score"] = best["f1_score"]
    info["roc_auc"] = best["roc_auc"]
    model_registry.save_registry_info(info)
    joblib_path = model_registry.export_best_model_copy(info["name"], info["alias"])
    print(f"Copy of best model saved to {joblib_path}")
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-only", action="store_true",
                        help="skip training; use runs already in mlflow.db")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("alembic").setLevel(logging.WARNING)

    try:
        setup_mlflow()
        df = data_preprocessing.load_dataset()
        csv = data_preprocessing.save_dataset_csv(df)
        print(f"Dataset: {df.shape[0]} rows x {df.shape[1] - 1} features -> {csv}")
        data = data_preprocessing.split_data(df, test_size=args.test_size)

        if not args.registry_only:
            run_training(data)
        run_registry_demo(data)
    except Exception as exc:  # one clear message instead of a wall of traceback
        log.error("Pipeline failed: %s", exc)
        return 1

    # Relative URI on purpose: absolute Windows paths with spaces/brackets break the shell.
    print("\nDone. Open a NEW terminal in this folder and run:\n"
          "    mlflow ui --backend-store-uri sqlite:///mlflow.db\n"
          "then open http://127.0.0.1:5000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
