"""Central configuration. Every tunable value lives here (reproducibility)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

# --- MLflow ---------------------------------------------------------------
# A SQLite database is the backend store. It is required for the Model Registry
# and works on Windows with no extra installation.
DB_PATH = PROJECT_ROOT / "mlflow.db"
# as_posix() gives forward slashes, which SQLAlchemy needs on Windows too.
TRACKING_URI = "sqlite:///" + DB_PATH.as_posix()
# Where MLflow stores artifacts (plots, reports, models) for new experiments.
ARTIFACT_ROOT = ARTIFACTS_DIR / "mlflow_artifacts"

EXPERIMENT_NAME = "MLflow_Experiment_Tracking"
REGISTERED_MODEL_NAME = "Best_Classification_Model"
# Model aliases replace the deprecated "Staging/Production" stages (MLflow 2.9+).
MODEL_ALIAS = "champion"

# --- Data / reproducibility -----------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_NAME = "breast_cancer_wisconsin"

# --- Model selection ------------------------------------------------------
PRIMARY_METRIC = "f1_score"     # first sort key
SECONDARY_METRIC = "roc_auc"    # tie-breaker

# --- Hyperparameters (one dict per experiment) -----------------------------
LOGISTIC_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
}
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_split": 2,
}
