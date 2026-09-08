from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .model import feature_columns, make_classifier


@dataclass
class EvaluationResult:
    modality: str
    accuracy: float
    f1: float
    auc: float | None
    n_rows: int
    n_participants: int


def evaluate_grouped(df: pd.DataFrame, modality: str, target_col: str = "label") -> EvaluationResult:
    required = {"participant_id", target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cols = feature_columns(df, modality)
    if not cols:
        raise ValueError(f"No feature columns found for modality: {modality}")

    work = df.dropna(subset=[target_col, "participant_id"]).copy()
    groups = work["participant_id"].astype(str)
    y = work[target_col].astype(int)

    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two participants are required for participant-independent evaluation.")

    n_splits = min(5, unique_groups)
    cv = GroupKFold(n_splits=n_splits)
    truth, pred, prob = [], [], []

    for train_idx, test_idx in cv.split(work[cols], y, groups):
        model = make_classifier()
        model.fit(work.iloc[train_idx][cols], y.iloc[train_idx])
        pred.extend(model.predict(work.iloc[test_idx][cols]))
        try:
            prob.extend(model.predict_proba(work.iloc[test_idx][cols])[:, 1])
        except Exception:
            prob.extend([float(p) for p in model.predict(work.iloc[test_idx][cols])])
        truth.extend(y.iloc[test_idx])

    auc = None
    try:
        auc = float(roc_auc_score(truth, prob))
    except ValueError:
        pass

    return EvaluationResult(
        modality=modality,
        accuracy=float(accuracy_score(truth, pred)),
        f1=float(f1_score(truth, pred, zero_division=0)),
        auc=auc,
        n_rows=len(work),
        n_participants=unique_groups,
    )
