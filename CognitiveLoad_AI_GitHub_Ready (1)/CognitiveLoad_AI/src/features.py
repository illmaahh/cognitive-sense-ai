from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _safe_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a / (b + eps))


def face_features(landmarks: Optional[Iterable]) -> dict[str, float]:
    """Facial-behavior proxies from MediaPipe Face Mesh landmarks."""
    if landmarks is None:
        return {}
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
    try:
        left_eye_v = _dist((pts[159][0], pts[159][1]), (pts[145][0], pts[145][1]))
        left_eye_h = _dist((pts[33][0], pts[33][1]), (pts[133][0], pts[133][1]))
        right_eye_v = _dist((pts[386][0], pts[386][1]), (pts[374][0], pts[374][1]))
        right_eye_h = _dist((pts[362][0], pts[362][1]), (pts[263][0], pts[263][1]))
        mouth_v = _dist((pts[13][0], pts[13][1]), (pts[14][0], pts[14][1]))
        mouth_h = _dist((pts[78][0], pts[78][1]), (pts[308][0], pts[308][1]))
        return {
            "face_left_eye_open": _safe_ratio(left_eye_v, left_eye_h),
            "face_right_eye_open": _safe_ratio(right_eye_v, right_eye_h),
            "face_mouth_open": _safe_ratio(mouth_v, mouth_h),
        }
    except (IndexError, TypeError):
        return {}


def gaze_features(landmarks: Optional[Iterable]) -> dict[str, float]:
    """Approximate gaze direction from iris-to-eye-corner normalized offsets."""
    if landmarks is None:
        return {}
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
    try:
        left_iris_x = np.mean([pts[i][0] for i in (468, 469, 470, 471)])
        right_iris_x = np.mean([pts[i][0] for i in (473, 474, 475, 476)])
        left_min, left_max = pts[33][0], pts[133][0]
        right_min, right_max = pts[362][0], pts[263][0]
        left_g = _safe_ratio(left_iris_x - min(left_min, left_max), abs(left_max - left_min))
        right_g = _safe_ratio(right_iris_x - min(right_min, right_max), abs(right_max - right_min))
        return {
            "gaze_left_horizontal": left_g,
            "gaze_right_horizontal": right_g,
            "gaze_dispersion_proxy": abs(left_g - right_g),
        }
    except (IndexError, TypeError):
        return {}


def pose_features(pose_landmarks: Optional[Iterable]) -> dict[str, float]:
    if pose_landmarks is None:
        return {}
    pts = [(lm.x, lm.y, lm.z, lm.visibility) for lm in pose_landmarks]
    try:
        ls, rs = pts[11], pts[12]
        lh, rh = pts[23], pts[24]
        shoulder_width = _dist((ls[0], ls[1]), (rs[0], rs[1]))
        hip_width = _dist((lh[0], lh[1]), (rh[0], rh[1]))
        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        torso_lean = _safe_ratio(abs(shoulder_y - hip_y), shoulder_width)
        return {
            "pose_shoulder_width": shoulder_width,
            "pose_hip_width": hip_width,
            "pose_torso_lean": torso_lean,
            "pose_visibility": float(np.mean([ls[3], rs[3], lh[3], rh[3]])),
        }
    except (IndexError, TypeError):
        return {}


def performance_features(correct: int, reaction_time_ms: float, task_progress: float) -> dict[str, float]:
    return {
        "perf_correct": float(correct),
        "perf_reaction_time_s": float(reaction_time_ms) / 1000.0,
        "perf_progress": float(task_progress),
    }


def fuse_features(*blocks: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for block in blocks:
        merged.update(block)
    return merged


def heuristic_load_score(features: dict[str, float]) -> float:
    """Demo-only proxy score in [0, 100]; replace with validated participant-trained model."""
    if not features:
        return 0.0

    score = 50.0
    rt = features.get("perf_reaction_time_s")
    gd = features.get("gaze_dispersion_proxy")
    acc = features.get("perf_correct")
    lean = features.get("pose_torso_lean")

    if rt is not None:
        score += min(max(rt - 1.0, -1.0), 2.0) * 10
    if gd is not None:
        score += min(max(gd, 0.0), 0.5) * 25
    if acc is not None:
        score += (0.5 - acc) * 20
    if lean is not None:
        score += min(max(lean, 0.0), 1.0) * 5

    return float(np.clip(score, 0, 100))
