"""
CBAS v2 — session_manager.py
Per-session context + pipeline runner + report generator.
Imports: schemas.py, feature_engineering.py, cognitive_model.py,
         reasoning_engine.py  — all same folder.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np

from schemas import PerceptionFrame, AnalyzeResponse, SessionReport
from feature_engineering import CircularBuffer, FeatureEngineeringModule
from cognitive_model import CognitiveStateModel
from reasoning_engine import (
    ReasoningEngine, AnomalyDetector, PredictiveModel,
    BehavioralScorer, InsightGenerator,
)


# ─────────────────────────────────────────────
# SESSION CONTEXT
# ─────────────────────────────────────────────

@dataclass
class SessionContext:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    buffer:          CircularBuffer       = field(default_factory=lambda: CircularBuffer(600))
    feature_module:  FeatureEngineeringModule = field(init=False)
    cognitive_model: CognitiveStateModel  = field(default_factory=CognitiveStateModel)
    reasoning:       ReasoningEngine      = field(default_factory=ReasoningEngine)
    anomaly_det:     AnomalyDetector      = field(default_factory=AnomalyDetector)
    predictor:       PredictiveModel      = field(default_factory=PredictiveModel)
    scorer:          BehavioralScorer     = field(default_factory=BehavioralScorer)
    insight_gen:     InsightGenerator     = field(default_factory=InsightGenerator)

    focus_hist:  deque = field(default_factory=lambda: deque(maxlen=7200))
    stress_hist: deque = field(default_factory=lambda: deque(maxlen=7200))
    load_hist:   deque = field(default_factory=lambda: deque(maxlen=7200))
    state_hist:  deque = field(default_factory=lambda: deque(maxlen=7200))

    transitions: list = field(default_factory=list)
    last_state:  str  = "idle"
    anomaly_log: list = field(default_factory=list)
    frame_count: int  = 0

    def __post_init__(self):
        self.feature_module = FeatureEngineeringModule(self.buffer)

    def touch(self):
        self.last_active = time.time()

    def is_expired(self, ttl: float = 1800.0) -> bool:
        return (time.time() - self.last_active) > ttl

    def record_state(self, s: str, ts: float):
        self.state_hist.append(s)
        if s != self.last_state:
            self.transitions.append({"from": self.last_state, "to": s,
                                      "timestamp": ts, "frame": self.frame_count})
            self.last_state = s

    def record_anomaly(self, a):
        if a.anomaly:
            self.anomaly_log.append({
                "type": a.type, "severity": a.severity,
                "z_score": a.z_score, "timestamp": a.timestamp,
                "frame": self.frame_count,
            })


# ─────────────────────────────────────────────
# SESSION REGISTRY
# ─────────────────────────────────────────────

class SessionRegistry:
    def __init__(self):
        self._store: dict = {}

    def create(self, session_id: Optional[str] = None) -> SessionContext:
        sid = session_id or str(uuid.uuid4())
        ctx = SessionContext(session_id=sid)
        self._store[sid] = ctx
        return ctx

    def get(self, sid: str) -> Optional[SessionContext]:
        return self._store.get(sid)

    def get_or_create(self, sid: str) -> SessionContext:
        return self._store.get(sid) or self.create(sid)

    def purge_expired(self):
        for sid in [s for s, c in self._store.items() if c.is_expired()]:
            del self._store[sid]

    def active_count(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline(ctx: SessionContext, frame: PerceptionFrame) -> AnalyzeResponse:
    """Full 4-layer analysis for one frame. ~0.5 ms."""
    import time as _t
    t0 = _t.perf_counter()

    ctx.frame_count += 1
    ctx.touch()
    ctx.buffer.push(frame)

    tf         = ctx.feature_module.extract(30)
    long_trend = ctx.feature_module.extract_long_trend()
    cog        = ctx.cognitive_model.infer(frame, tf)
    anomaly    = ctx.anomaly_det.update_and_detect(frame, tf)
    ctx.predictor.update(cog.dominant_state)
    pred       = ctx.predictor.predict(cog.dominant_state, tf)
    expl       = ctx.reasoning.explain(frame, tf, cog, long_trend)
    score      = ctx.scorer.score(cog, tf)

    ctx.focus_hist.append(cog.focus_score)
    ctx.stress_hist.append(cog.stress_level)
    ctx.load_hist.append(cog.cognitive_load)
    ctx.record_state(cog.dominant_state, frame.frame_ts)
    ctx.record_anomaly(anomaly)

    ms = round((_t.perf_counter() - t0) * 1000, 2)
    return AnalyzeResponse(
        session_id=frame.session_id, timestamp=frame.frame_ts,
        cognitive_state=cog, explanation=expl, anomaly=anomaly,
        prediction=pred, behavioral_score=score,
        temporal_features=tf, processing_ms=ms,
    )


# ─────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────

def build_report(ctx: SessionContext) -> SessionReport:
    now      = time.time()
    duration = now - ctx.created_at

    fa  = np.array(list(ctx.focus_hist))  if ctx.focus_hist  else np.array([0.0])
    sa  = np.array(list(ctx.stress_hist)) if ctx.stress_hist else np.array([0.0])
    la  = np.array(list(ctx.load_hist))   if ctx.load_hist   else np.array([0.0])

    states   = list(ctx.state_hist)
    total    = max(1, len(states))
    all_s    = ["focused","distracted","stressed","fatigued","overloaded","idle","engaged"]
    dom_pct  = {s: round(states.count(s)/total*100, 1) for s in all_s}

    avg_f  = round(float(fa.mean()), 1)
    avg_st = round(float(sa.mean()), 1)
    avg_ld = round(float(la.mean()), 1)
    calm   = max(0, 100 - avg_st)
    overal = round(avg_f*0.4 + calm*0.3 + (100-avg_ld)*0.3, 1)
    grade  = next(g for t, g in [(90,"A"),(75,"B"),(60,"C"),(40,"D"),(0,"F")] if overal >= t)

    def ds(arr, n=120):
        if len(arr) <= n:
            return [round(float(v),1) for v in arr]
        idx = [int(i) for i in np.linspace(0, len(arr)-1, n)]
        return [round(float(arr[i]),1) for i in idx]

    dom_state = max(dom_pct, key=dom_pct.get)
    insights  = [
        f"Primary state: {dom_state} ({dom_pct[dom_state]}% of session).",
        "High sustained focus." if avg_f > 75 else "Low average focus — attention fragmentation." if avg_f < 40 else f"Moderate focus ({avg_f}%).",
    ]
    if len(ctx.anomaly_log) > 2:
        insights.append(f"{len(ctx.anomaly_log)} anomalies detected.")
    insights.append(f"Session: {duration/60:.1f} min, {ctx.frame_count} frames analyzed.")

    rec = ("Excellent session." if avg_f > 70
           else "Elevated stress — review environment." if avg_st > 60
           else "Low focus — identify distraction sources." if avg_f < 40
           else "Use Pomodoro technique to improve sustained focus.")

    return SessionReport(
        session_id=ctx.session_id, start_time=ctx.created_at,
        end_time=now, duration_seconds=round(duration, 1),
        total_frames=ctx.frame_count,
        avg_focus=avg_f, avg_cognitive_load=avg_ld,
        avg_stress=avg_st, avg_fatigue=0.0,
        overall_score=overal, grade=grade,
        state_transitions=ctx.transitions[-50:],
        dominant_state_pct=dom_pct,
        anomaly_count=len(ctx.anomaly_log),
        anomaly_log=ctx.anomaly_log[-20:],
        focus_timeline=ds(fa), stress_timeline=ds(sa), load_timeline=ds(la),
        key_insights=insights, final_recommendation=rec,
    )
