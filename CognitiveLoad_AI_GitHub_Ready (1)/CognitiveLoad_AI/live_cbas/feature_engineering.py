"""
CBAS v2 — feature_engineering.py
Layer 2: Sliding-window temporal feature extraction.
Imports only: schemas.py (same folder), numpy (stdlib available).
"""
from __future__ import annotations
import math
import time
from collections import deque
from typing import Optional

import numpy as np

# Same-folder import
from schemas import PerceptionFrame, TemporalFeatures


# ─────────────────────────────────────────────
# CIRCULAR BUFFER
# ─────────────────────────────────────────────

class CircularBuffer:
    """Fixed-capacity ring buffer for streaming PerceptionFrames."""

    def __init__(self, capacity: int = 600):
        self.capacity = capacity
        self._frames: deque = deque(maxlen=capacity)

    def push(self, frame: PerceptionFrame) -> None:
        self._frames.append(frame)

    def __len__(self) -> int:
        return len(self._frames)

    def last_n(self, n: int) -> list:
        frames = list(self._frames)
        return frames[-n:] if n <= len(frames) else frames

    def last_n_seconds(self, seconds: float) -> list:
        now = time.time() * 1000
        cutoff = now - seconds * 1000
        return [f for f in self._frames if f.frame_ts >= cutoff]

    def is_ready(self, min_frames: int = 5) -> bool:
        return len(self._frames) >= min_frames


# ─────────────────────────────────────────────
# STATISTICAL HELPERS
# ─────────────────────────────────────────────

def _slope(values: np.ndarray) -> float:
    """Least-squares slope per index step."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), values.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (values - ym)).sum() / denom) if denom else 0.0


def _entropy(values: np.ndarray, bins: int = 10) -> float:
    """Shannon entropy of value distribution."""
    if len(values) < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0].astype(float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-9)).sum())


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation (std/mean)."""
    m = values.mean()
    return float(values.std() / (m + 1e-9)) if m else 0.0


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────
# FEATURE ENGINEERING MODULE
# ─────────────────────────────────────────────

class FeatureEngineeringModule:
    """
    Extracts 13 temporal features from a sliding window of frames.
    Primary window: 30 seconds.
    Long-trend window: 120 seconds.
    """

    FPS = 15.0  # assumed camera FPS for rate calculations

    def __init__(self, buffer: CircularBuffer):
        self.buffer = buffer

    # ── Main extraction ──────────────────────

    def extract(self, window_s: int = 30) -> TemporalFeatures:
        frames = self.buffer.last_n_seconds(window_s)
        if len(frames) < 5:
            return TemporalFeatures(window_seconds=window_s)

        gaze_x   = np.array([f.gaze_x          for f in frames])
        gaze_y   = np.array([f.gaze_y          for f in frames])
        eye_stab = np.array([f.eye_stability_raw for f in frames])
        blink_r  = np.array([f.blink_rate       for f in frames])
        hand_vel = np.array([f.hand_velocity    for f in frames])
        body_mot = np.array([f.body_motion      for f in frames])
        attention= np.array([f.attention_raw    for f in frames])
        shoulder = np.array([f.shoulder_level   for f in frames])
        spine    = np.array([f.spine_angle      for f in frames])

        # Gaze
        gaze_mag  = np.sqrt(gaze_x**2 + gaze_y**2)
        gaze_var  = float(np.var(gaze_mag))
        if len(gaze_x) > 1:
            dx = np.diff(gaze_x); dy = np.diff(gaze_y)
            drift = float(np.sqrt(dx**2 + dy**2).mean() * self.FPS)
        else:
            drift = 0.0

        # Motion
        all_motion = np.concatenate([hand_vel, body_mot])
        entropy    = _entropy(all_motion)
        micro_ct   = int(np.sum((hand_vel > 0.1) & (hand_vel < 1.5)))

        # Posture
        shoulder_pen = _clamp(float(shoulder.mean()) * 3, 0, 50)
        spine_pen    = _clamp(float(np.abs(spine).mean()) * 1.5, 0, 50)
        posture_stab = max(0.0, 100.0 - shoulder_pen - spine_pen)

        # Trend slope (per minute)
        att_slope   = _slope(attention) * self.FPS * 60
        stab_slope  = _slope(eye_stab)  * self.FPS

        return TemporalFeatures(
            window_seconds=window_s,
            gaze_variance=round(gaze_var, 5),
            gaze_drift_rate=round(drift, 4),
            eye_stability_mean=round(float(eye_stab.mean()), 2),
            eye_stability_trend=round(stab_slope, 4),
            blink_freq_mean=round(float(blink_r.mean()), 2),
            blink_irregularity=round(_cv(blink_r), 4),
            velocity_mean=round(float(hand_vel.mean()), 3),
            velocity_variance=round(float(np.var(hand_vel)), 4),
            micro_movement_count=micro_ct,
            motion_entropy=round(entropy, 4),
            attention_mean=round(float(attention.mean()), 2),
            attention_variance=round(float(np.var(attention)), 4),
            attention_slope=round(att_slope, 4),
            posture_stability=round(posture_stab, 2),
        )

    # ── Long-trend (120s) ────────────────────

    def extract_long_trend(self) -> dict:
        frames = self.buffer.last_n_seconds(120)
        if len(frames) < 20:
            return {"direction": "stable", "slope_per_min": 0.0,
                    "cyclical_distraction": False, "distraction_transitions": 0,
                    "attention_mean_long": 0.0}

        attention = np.array([f.attention_raw for f in frames])
        slope = _slope(attention) * self.FPS * 60

        direction = "improving" if slope > 1.0 else "deteriorating" if slope < -1.0 else "stable"

        below = (attention < 40).astype(int)
        transitions = int(np.sum(np.abs(np.diff(below))))

        return {
            "direction": direction,
            "slope_per_min": round(float(slope), 3),
            "cyclical_distraction": transitions > 4,
            "distraction_transitions": transitions,
            "attention_mean_long": round(float(attention.mean()), 2),
        }
