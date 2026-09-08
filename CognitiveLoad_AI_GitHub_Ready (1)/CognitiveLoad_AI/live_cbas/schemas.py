"""
CBAS v2 — schemas.py
All data models. No external dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import time


@dataclass
class PerceptionFrame:
    session_id: str
    frame_ts: float
    face_detected: bool = False
    hands_detected: bool = False
    pose_detected: bool = False
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    eye_openness: float = 1.0
    blink_rate: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    hand_velocity: float = 0.0
    hand_activity: float = 0.0
    gesture: str = "IDLE"
    body_motion: float = 0.0
    shoulder_level: float = 0.0
    spine_angle: float = 0.0
    attention_raw: float = 0.0
    movement_raw: float = 0.0
    eye_stability_raw: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "PerceptionFrame":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalFeatures:
    window_seconds: int = 30
    gaze_variance: float = 0.0
    gaze_drift_rate: float = 0.0
    eye_stability_mean: float = 0.0
    eye_stability_trend: float = 0.0
    blink_freq_mean: float = 0.0
    blink_irregularity: float = 0.0
    velocity_mean: float = 0.0
    velocity_variance: float = 0.0
    micro_movement_count: int = 0
    motion_entropy: float = 0.0
    attention_mean: float = 0.0
    attention_variance: float = 0.0
    attention_slope: float = 0.0
    posture_stability: float = 50.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CognitiveState:
    focus_score: float = 0.0
    cognitive_load: float = 0.0
    stress_level: float = 0.0
    fatigue_level: float = 0.0
    engagement: float = 0.0
    confidence: float = 0.0
    dominant_state: str = "idle"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StateExplanation:
    state: str = ""
    reason: str = ""
    confidence: float = 0.0
    contributing_signals: list = field(default_factory=list)
    trend_note: Optional[str] = None
    recommendation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnomalyReport:
    anomaly: bool = False
    type: Optional[str] = None
    severity: float = 0.0
    z_score: Optional[float] = None
    affected_metric: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatePrediction:
    next_state: str = "idle"
    horizon_seconds: int = 30
    probability_distribution: dict = field(default_factory=dict)
    confidence: float = 0.0
    trend_direction: str = "stable"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehavioralScore:
    overall: float = 0.0
    components: dict = field(default_factory=dict)
    grade: str = "F"
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalyzeResponse:
    session_id: str = ""
    timestamp: float = 0.0
    cognitive_state: CognitiveState = field(default_factory=CognitiveState)
    explanation: StateExplanation = field(default_factory=StateExplanation)
    anomaly: AnomalyReport = field(default_factory=AnomalyReport)
    prediction: StatePrediction = field(default_factory=StatePrediction)
    behavioral_score: BehavioralScore = field(default_factory=BehavioralScore)
    temporal_features: TemporalFeatures = field(default_factory=TemporalFeatures)
    processing_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "cognitive_state": self.cognitive_state.to_dict(),
            "explanation": self.explanation.to_dict(),
            "anomaly": self.anomaly.to_dict(),
            "prediction": self.prediction.to_dict(),
            "behavioral_score": self.behavioral_score.to_dict(),
            "temporal_features": self.temporal_features.to_dict(),
            "processing_ms": self.processing_ms,
        }


@dataclass
class SessionReport:
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    avg_focus: float = 0.0
    avg_cognitive_load: float = 0.0
    avg_stress: float = 0.0
    avg_fatigue: float = 0.0
    overall_score: float = 0.0
    grade: str = "F"
    state_transitions: list = field(default_factory=list)
    dominant_state_pct: dict = field(default_factory=dict)
    anomaly_count: int = 0
    anomaly_log: list = field(default_factory=list)
    focus_timeline: list = field(default_factory=list)
    stress_timeline: list = field(default_factory=list)
    load_timeline: list = field(default_factory=list)
    key_insights: list = field(default_factory=list)
    final_recommendation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
