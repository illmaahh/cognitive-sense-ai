import pandas as pd

from src.model import feature_columns
from src.research import evaluate_grouped


def test_feature_columns_are_modality_specific():
    df = pd.DataFrame(
        {
            "face_left_eye_open": [0.1],
            "gaze_left_horizontal": [0.2],
            "pose_torso_lean": [0.3],
            "perf_correct": [1],
            "label": [0],
        }
    )
    assert feature_columns(df, "face") == ["face_left_eye_open"]
    assert feature_columns(df, "gaze") == ["gaze_left_horizontal"]


def test_grouped_evaluation_requires_multiple_participants():
    df = pd.DataFrame(
        {
            "participant_id": ["P1", "P1"],
            "face_left_eye_open": [0.1, 0.2],
            "label": [0, 1],
        }
    )
    try:
        evaluate_grouped(df, "face")
    except ValueError as exc:
        assert "two participants" in str(exc)
    else:
        raise AssertionError("Expected a participant-count error")
