"""Dataset loading, validation and train/test splitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)
TARGET_COLUMN = "target"  # 0 = malignant, 1 = benign (scikit-learn convention)


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_size: float
    random_state: int

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def load_dataset() -> pd.DataFrame:
    """Load Breast Cancer Wisconsin (569 rows, 30 features) as one DataFrame."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    if df.empty:
        raise ValueError("Dataset loaded empty")
    return df


def save_dataset_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the dataset to data/ so it can be inspected during the demo."""
    path = Path(path or config.DATA_DIR / "breast_cancer.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def validate_dataset(df: pd.DataFrame) -> None:
    """Fail early on bad data (missing target, NaNs, single class)."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'")
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target must contain at least two classes")


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
               random_state: int = config.RANDOM_STATE) -> DataSplit:
    """Stratified train/test split (keeps class balance identical in both parts)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    validate_dataset(df)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info("Split data: train=%d test=%d features=%d",
                len(X_tr), len(X_te), X.shape[1])
    return DataSplit(X_tr, X_te, y_tr, y_te, test_size, random_state)


def get_data(test_size: float = config.TEST_SIZE,
             random_state: int = config.RANDOM_STATE) -> DataSplit:
    """Convenience: load -> validate -> split."""
    return split_data(load_dataset(), test_size, random_state)
