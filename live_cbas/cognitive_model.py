"""
CognitiveSense AI v4 — cognitive_model.py
Layer 3: Infers latent cognitive states from temporal features.
Imports only: schemas.py (same folder), math (stdlib).
"""
from __future__ import annotations
import math

from schemas import PerceptionFrame, TemporalFeatures, CognitiveState


# ─────────────────────────────────────────────
# WEIGHT TABLES
# Each entry: signal_name -> (weight, invert)
# invert=True: high value HURTS this dimension
# ─────────────────────────────────────────────

FOCUS_W = {
    "eye_stability":      (0.30, False),
    "attention_raw":      (0.25, False),
    "gaze_variance":      (0.15, True),
    "body_motion":        (0.12, True),
    "head_yaw":           (0.10, True),
    "blink_regularity":   (0.08, False),
}

LOAD_W = {
    "attention_raw":      (0.20, False),
    "blink_freq":         (0.25, True),   # low blink = high load
    "gaze_variance":      (0.20, False),
    "velocity_variance":  (0.15, False),
    "motion_entropy":     (0.20, False),
}

STRESS_W = {
    "body_motion":        (0.25, False),
    "hand_velocity":      (0.20, False),
    "velocity_variance":  (0.15, False),
    "head_yaw":           (0.15, False),
    "gaze_drift_rate":    (0.15, False),
    "posture_stability":  (0.10, True),
}

FATIGUE_W = {
    "eye_openness":       (0.30, True),
    "blink_freq":         (0.20, False),  # high blink = fatigue
    "attention_slope":    (0.25, True),   # declining = fatigue
    "posture_stability":  (0.15, True),
    "motion_entropy":     (0.10, True),   # low entropy = lethargic
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _norm(value: float, lo: float, hi: float, invert: bool = False) -> float:
    r = hi - lo
    if r == 0:
        return 0.0
    n = _clamp01((value - lo) / r)
    return 1.0 - n if invert else n


def _sigmoid(x: float, center: float = 0.0, k: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - center)))


def _build_signals(frame: PerceptionFrame, tf: TemporalFeatures) -> dict:
    """Normalize all raw signals to [0, 1]."""
    blink_reg = _clamp01(1.0 - tf.blink_irregularity)
    return {
        "eye_stability":     frame.eye_stability_raw / 100.0,
        "attention_raw":     frame.attention_raw / 100.0,
        "body_motion":       frame.body_motion / 100.0,
        "hand_activity":     frame.hand_activity / 100.0,
        "posture_stability": tf.posture_stability / 100.0,
        "eye_openness":      _clamp01(frame.eye_openness),
        "hand_velocity":     _norm(frame.hand_velocity, 0, 30),
        "head_yaw":          _norm(abs(frame.head_yaw), 0, 45),
        "gaze_variance":     _norm(tf.gaze_variance, 0, 0.12),
        "gaze_drift_rate":   _norm(tf.gaze_drift_rate, 0, 5.0),
        "blink_freq":        _norm(tf.blink_freq_mean, 0, 40),
        "blink_regularity":  blink_reg,
        "velocity_mean":     _norm(tf.velocity_mean, 0, 20),
        "velocity_variance": _norm(tf.velocity_variance, 0, 50),
        "motion_entropy":    _norm(tf.motion_entropy, 0, 3.0),
        "attention_slope":   _norm(tf.attention_slope, -5, 5),
    }


def _score(signals: dict, weights: dict) -> float:
    """Weighted average → 0–100."""
    total_w = sum(w for w, _ in weights.values())
    if total_w == 0:
        return 0.0
    acc = 0.0
    for name, (w, inv) in weights.items():
        v = signals.get(name, 0.0)
        acc += w * (1.0 - v if inv else v)
    return _clamp01(acc / total_w) * 100.0


def _confidence(frame: PerceptionFrame, tf: TemporalFeatures) -> float:
    mod = (
        (0.40 if frame.face_detected else 0.0)
        + (0.30 if frame.pose_detected else 0.0)
        + (0.15 if frame.hands_detected else 0.0)
        + 0.15
    )
    contradiction = abs((frame.attention_raw / 100.0) - (1.0 - frame.body_motion / 100.0))
    data_ok = _clamp01(tf.window_seconds / 30.0) * 0.10
    return _clamp01(mod - contradiction * 0.2 + data_ok)


def _dominant(focus, load, stress, fatigue, engagement, face: bool) -> str:
    if not face:
        return "idle"
    m = {
        "idle":       0.0,
        "focused":    focus  * (1.0 - stress / 200.0) * (1.0 - fatigue / 200.0) / 100.0,
        "distracted": (1.0 - focus / 100.0) * (1.0 - stress / 150.0),
        "stressed":   stress  / 100.0 * (1.0 - fatigue / 200.0),
        "fatigued":   fatigue / 100.0 * (1.0 - stress / 200.0),
        "overloaded": (load / 100.0) * (stress / 200.0 + 0.5),
        "engaged":    engagement / 100.0 * (1.0 - fatigue / 200.0) * (1.0 - stress / 200.0),
    }
    return max(m, key=m.get)


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class CognitiveStateModel:
    """
    Probabilistic cognitive state inference with EMA smoothing.
    Maintains running averages so state doesn't jitter frame-to-frame.
    """
    ALPHA = 0.20

    def __init__(self):
        self._ema = dict(focus=50.0, load=20.0, stress=20.0, fatigue=20.0, engagement=50.0)

    def infer(self, frame: PerceptionFrame, tf: TemporalFeatures) -> CognitiveState:
        sigs = _build_signals(frame, tf)

        raw_focus   = _score(sigs, FOCUS_W)
        raw_load    = _score(sigs, LOAD_W)
        raw_stress  = _score(sigs, STRESS_W)
        raw_fatigue = _score(sigs, FATIGUE_W)
        raw_engage  = raw_focus * 0.5 + (100 - raw_stress) * 0.3 + (100 - raw_fatigue) * 0.2

        for k, raw in [("focus", raw_focus), ("load", raw_load),
                       ("stress", raw_stress), ("fatigue", raw_fatigue),
                       ("engagement", raw_engage)]:
            self._ema[k] = self.ALPHA * raw + (1 - self.ALPHA) * self._ema[k]

        conf = _confidence(frame, tf)
        dom  = _dominant(self._ema["focus"], self._ema["load"], self._ema["stress"],
                         self._ema["fatigue"], self._ema["engagement"], frame.face_detected)

        return CognitiveState(
            focus_score    = round(self._ema["focus"],      1),
            cognitive_load = round(self._ema["load"],       1),
            stress_level   = round(self._ema["stress"],     1),
            fatigue_level  = round(self._ema["fatigue"],    1),
            engagement     = round(self._ema["engagement"], 1),
            confidence     = round(conf, 3),
            dominant_state = dom,
        )
