from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODALITY_PREFIXES = {
    "face": ("face_",),
    "gaze": ("gaze_",),
    "pose": ("pose_",),
    "performance": ("perf_",),
}

META_COLUMNS = {"participant_id", "session_id", "timestamp", "label", "task", "difficulty"}


def feature_columns(df: pd.DataFrame, modality: str | None = None) -> list[str]:
    if modality is None:
        prefixes = tuple(p for group in MODALITY_PREFIXES.values() for p in group)
    else:
        prefixes = MODALITY_PREFIXES[modality]
    return [c for c in df.columns if c.startswith(prefixes) and c not in META_COLUMNS]


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )


def train_model(df: pd.DataFrame, feature_cols: Iterable[str], target_col: str = "label") -> Pipeline:
    x = df[list(feature_cols)]
    y = df[target_col].astype(int)
    model = make_classifier()
    model.fit(x, y)
    return model


def predict_proba(model: Pipeline, frame: pd.DataFrame, feature_cols: Iterable[str]) -> np.ndarray:
    return model.predict_proba(frame[list(feature_cols)])[:, 1]


def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
