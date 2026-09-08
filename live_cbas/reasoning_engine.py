"""
CBAS v2 — reasoning_engine.py
Layer 4: Reasoning, Anomaly Detection, Prediction, Scoring, Insights.
Imports only: schemas.py (same folder), numpy, collections, math.
"""
from __future__ import annotations
import math
import time
from collections import deque
from typing import Optional

import numpy as np

from schemas import (
    PerceptionFrame, TemporalFeatures, CognitiveState,
    StateExplanation, AnomalyReport, StatePrediction, BehavioralScore,
)


# ══════════════════════════════════════════
# 4a — REASONING ENGINE
# ══════════════════════════════════════════

class ReasoningEngine:

    _RECS = {
        "focused":    "Maintain current environment. Avoid interruptions — you are in a productive state.",
        "distracted": "Remove visual distractors. Try the 2-minute rule: finish one small task completely before switching.",
        "stressed":   "Take a 5-minute break. Practice box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
        "fatigued":   "Consider a 15-20 min break or a walk. Hydration and posture reset may help immediately.",
        "overloaded": "Decompose your current task. Work in 25-min Pomodoro intervals with 5-min breaks.",
        "idle":       "System is waiting for presence. Position yourself in front of the camera.",
        "engaged":    "High engagement. Tackle your most demanding work now.",
    }

    def explain(self, frame: PerceptionFrame, tf: TemporalFeatures,
                state: CognitiveState, long_trend: dict) -> StateExplanation:
        signals   = self._attribute(frame, tf, state)
        reason    = self._reason(state.dominant_state, signals)
        trend_note= self._trend(long_trend)
        rec       = self._RECS.get(state.dominant_state, "Continue monitoring.")
        return StateExplanation(
            state=state.dominant_state.replace("_", " ").title(),
            reason=reason,
            confidence=state.confidence,
            contributing_signals=signals,
            trend_note=trend_note,
            recommendation=rec,
        )

    def _attribute(self, frame, tf, state) -> list:
        out = []
        if tf.gaze_variance > 0.05:
            out.append(f"high gaze variance ({tf.gaze_variance:.4f})")
        if frame.hand_velocity > 8:
            out.append(f"rapid hand movement ({frame.hand_velocity:.1f})")
        if frame.body_motion > 60:
            out.append(f"elevated body motion ({frame.body_motion:.0f}%)")
        if frame.eye_openness < 0.4:
            out.append(f"reduced eye openness ({frame.eye_openness:.2f})")
        if tf.blink_freq_mean > 25:
            out.append(f"high blink rate ({tf.blink_freq_mean:.0f}/min — fatigue indicator)")
        elif tf.blink_freq_mean < 8 and frame.face_detected:
            out.append(f"suppressed blink rate ({tf.blink_freq_mean:.0f}/min — high load)")
        if frame.attention_raw < 35:
            out.append(f"low attention signal ({frame.attention_raw:.0f}%)")
        if frame.eye_stability_raw < 40:
            out.append(f"unstable eye fixation ({frame.eye_stability_raw:.0f}%)")
        if tf.posture_stability < 50:
            out.append(f"poor posture ({tf.posture_stability:.0f}%)")
        if state.focus_score > 70:
            out.append(f"strong focus signal ({state.focus_score:.0f}/100)")
        if state.engagement > 75:
            out.append(f"high engagement ({state.engagement:.0f}/100)")
        return out[:4] if out else [f"composite signal: {state.dominant_state}"]

    def _reason(self, dom: str, signals: list) -> str:
        prefix = {
            "focused":    "Consistent gaze fixation and low motor activity indicate",
            "distracted": "Attention fragmentation detected —",
            "stressed":   "Elevated autonomic arousal pattern —",
            "fatigued":   "Psychomotor slowing and ocular fatigue indicators —",
            "overloaded": "Effortful cognitive processing signature —",
            "idle":       "No subject detected in frame.",
            "engaged":    "Active, purposeful cognitive engagement indicated by",
        }.get(dom, "Signal pattern indicates")
        return f"{prefix} {'; '.join(signals[:3])}."

    def _trend(self, lt: dict) -> Optional[str]:
        if lt.get("direction") == "stable":
            return None
        notes = []
        if lt.get("direction") == "deteriorating":
            notes.append(f"Focus declining ~{abs(lt.get('slope_per_min',0)):.1f} pts/min.")
        elif lt.get("direction") == "improving":
            notes.append("Focus improving over the session.")
        if lt.get("cyclical_distraction"):
            notes.append("Cyclical distraction pattern detected.")
        return " ".join(notes) if notes else None


# ══════════════════════════════════════════
# 4b — ANOMALY DETECTOR
# ══════════════════════════════════════════

class AnomalyDetector:
    """Z-score anomaly detection over rolling 120-sample baseline."""
    WINDOW    = 120
    Z_THRESH  = 2.5
    SEV_SCALE = 5.0

    def __init__(self):
        self._bufs = {
            "attention_raw": deque(maxlen=self.WINDOW),
            "body_motion":   deque(maxlen=self.WINDOW),
            "hand_velocity": deque(maxlen=self.WINDOW),
            "gaze_var":      deque(maxlen=self.WINDOW),
        }

    def update_and_detect(self, frame: PerceptionFrame,
                          tf: TemporalFeatures) -> AnomalyReport:
        self._bufs["attention_raw"].append(frame.attention_raw)
        self._bufs["body_motion"].append(frame.body_motion)
        self._bufs["hand_velocity"].append(frame.hand_velocity)
        self._bufs["gaze_var"].append(tf.gaze_variance)

        if len(self._bufs["attention_raw"]) < 20:
            return AnomalyReport(anomaly=False, timestamp=time.time())

        checks = [
            ("sudden attention drop",    "attention_raw", frame.attention_raw, True),
            ("movement spike",           "body_motion",   frame.body_motion,   False),
            ("rapid hand motion",        "hand_velocity", frame.hand_velocity, False),
            ("gaze instability burst",   "gaze_var",      tf.gaze_variance,    False),
        ]

        best: Optional[tuple] = None
        for label, key, val, low_bad in checks:
            arr = np.array(self._bufs[key])
            mean, std = float(arr.mean()), float(arr.std())
            if std < 1e-6:
                continue
            z = (val - mean) / std
            if low_bad:
                z = -z
            if z > self.Z_THRESH:
                if best is None or z > best[1]:
                    best = (label, z)

        if best:
            label, z = best
            sev = min(1.0, (z - self.Z_THRESH) / self.SEV_SCALE)
            return AnomalyReport(
                anomaly=True, type=label,
                severity=round(sev, 3),
                z_score=round(z, 2),
                affected_metric=label,
                timestamp=time.time(),
            )
        return AnomalyReport(anomaly=False, timestamp=time.time())


# ══════════════════════════════════════════
# 4c — PREDICTIVE MODEL (Markov + slope bias)
# ══════════════════════════════════════════

class PredictiveModel:
    STATES   = ["focused","distracted","stressed","fatigued","overloaded","idle","engaged"]
    HORIZON  = 30

    def __init__(self):
        n = len(self.STATES)
        self._mat = np.ones((n, n)) * 0.1   # Laplace smoothing
        self._hist: deque = deque(maxlen=200)
        self._idx  = {s: i for i, s in enumerate(self.STATES)}

    def update(self, state: str) -> None:
        if self._hist and state in self._idx and self._hist[-1] in self._idx:
            self._mat[self._idx[self._hist[-1]], self._idx[state]] += 1
        self._hist.append(state)

    def predict(self, state: str, tf: TemporalFeatures) -> StatePrediction:
        if state not in self._idx or len(self._hist) < 5:
            p = {s: round(1/len(self.STATES), 3) for s in self.STATES}
            return StatePrediction(next_state=state, horizon_seconds=self.HORIZON,
                                   probability_distribution=p,
                                   confidence=round(1/len(self.STATES),3),
                                   trend_direction="stable")

        row = self._mat[self._idx[state]].copy()
        row /= row.sum()

        slope = tf.attention_slope
        if slope > 2.0:
            fi = self._idx["focused"]
            row[fi] = min(1.0, row[fi] * 1.5)
            row /= row.sum()
        elif slope < -2.0:
            di = self._idx["distracted"]
            row[di] = min(1.0, row[di] * 1.5)
            row /= row.sum()

        trend = "improving" if slope > 1.5 else "deteriorating" if slope < -1.5 else "stable"
        next_s = self.STATES[int(np.argmax(row))]
        prob   = {s: round(float(row[i]), 3) for i, s in enumerate(self.STATES)}
        return StatePrediction(
            next_state=next_s, horizon_seconds=self.HORIZON,
            probability_distribution=prob,
            confidence=round(float(row.max()), 3),
            trend_direction=trend,
        )


# ══════════════════════════════════════════
# 4d — BEHAVIORAL SCORER
# ══════════════════════════════════════════

class BehavioralScorer:
    _W = dict(focus=0.35, calmness=0.25, engagement=0.20, posture=0.10, alertness=0.10)
    _GRADES = [(90,"A","Optimal"),(75,"B","Good — sustained performance"),
               (60,"C","Moderate"),(40,"D","Below par"),(0,"F","Critical")]

    def score(self, state: CognitiveState, tf: TemporalFeatures) -> BehavioralScore:
        comps = {
            "focus":     round(state.focus_score, 1),
            "calmness":  round(max(0.0, 100 - state.stress_level), 1),
            "engagement":round(state.engagement, 1),
            "posture":   round(tf.posture_stability, 1),
            "alertness": round(max(0.0, 100 - state.fatigue_level), 1),
        }
        overall = round(max(0.0, min(100.0, sum(self._W[k]*v for k,v in comps.items()))), 1)
        grade, label = next((g, l) for t, g, l in self._GRADES if overall >= t)
        return BehavioralScore(overall=overall, components=comps, grade=grade, label=label)


# ══════════════════════════════════════════
# 4e — INSIGHT GENERATOR
# ══════════════════════════════════════════

class InsightGenerator:
    """Deterministic multi-sentence insight. No LLM / API key needed."""

    def generate(self, state: CognitiveState, expl: StateExplanation,
                 pred: StatePrediction, score: BehavioralScore,
                 long_trend: dict) -> str:
        s   = state.dominant_state
        f   = round(state.focus_score)
        ld  = round(state.cognitive_load)
        st  = round(state.stress_level)
        fa  = round(state.fatigue_level)
        eng = round(state.engagement)

        descs = {
            "focused":    f"High-quality focus state detected (focus {f}/100, load {ld}/100).",
            "distracted": f"Attention fragmentation active — focus has dropped to {f}/100.",
            "stressed":   f"Elevated stress signature (stress {st}/100, focus {f}/100).",
            "fatigued":   f"Fatigue indicators present (fatigue {fa}/100, alertness declining).",
            "overloaded": f"High cognitive load ({ld}/100) with stress signals ({st}/100).",
            "idle":       "No behavioral signal — awaiting camera presence.",
            "engaged":    f"Strong engagement state (engagement {eng}/100, focus {f}/100).",
        }
        parts = [descs.get(s, f"State: {s}.")]

        d = long_trend.get("direction", "stable")
        sl = abs(long_trend.get("slope_per_min", 0.0))
        if d == "deteriorating" and sl > 1.0:
            parts.append(f"Focus declining ~{sl:.1f} pts/min over last 2 minutes.")
        elif long_trend.get("cyclical_distraction"):
            parts.append("Repeated distraction cycles detected — task-switching disrupting deep work.")
        elif d == "improving":
            parts.append("Positive trend: focus improving steadily.")

        np_ = pred.next_state
        nc  = round(pred.confidence * 100)
        if np_ != s:
            parts.append(f"Prediction: transition to '{np_}' in ~{pred.horizon_seconds}s (conf {nc}%).")

        parts.append(f"Overall score: {score.overall}/100 ({score.grade} — {score.label}).")
        parts.append(expl.recommendation)
        return " ".join(parts)
