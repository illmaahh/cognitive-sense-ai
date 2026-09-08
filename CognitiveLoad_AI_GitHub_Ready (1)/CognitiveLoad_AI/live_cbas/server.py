"""
CBAS v2 — server.py
Self-contained API server. No folder structure needed.
Run:  python server.py
      python server.py --port 8080

All files must be in the SAME folder:
  server.py, schemas.py, feature_engineering.py,
  cognitive_model.py, reasoning_engine.py, session_manager.py
"""
from __future__ import annotations
import sys
import os
import json
import time
import uuid
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

# ── Make sure Python finds sibling modules ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Import CBAS modules ──────────────────────────────────────────────
from schemas import PerceptionFrame
from session_manager import SessionRegistry, run_pipeline, build_report
from reasoning_engine import ReasoningEngine, BehavioralScorer, InsightGenerator

# ── Global state ─────────────────────────────────────────────────────
registry    = SessionRegistry()
insight_gen = InsightGenerator()

# Background cleanup thread (every 5 min)
def _cleanup():
    while True:
        time.sleep(300)
        registry.purge_expired()

threading.Thread(target=_cleanup, daemon=True).start()


# ════════════════════════════════════════════════════════════════
# ROUTE HANDLERS  (pure logic, no HTTP coupling)
# ════════════════════════════════════════════════════════════════

def route_health() -> dict:
    return {"status": "operational", "version": "3.0.0",
            "active_sessions": registry.active_count(),
            "timestamp": time.time()}


def route_analyze(body: dict) -> dict:
    fd = body.get("frame", body)
    fd.setdefault("session_id", str(uuid.uuid4()))
    fd.setdefault("frame_ts", time.time() * 1000)
    frame = PerceptionFrame.from_dict(fd)
    ctx   = registry.get_or_create(frame.session_id)
    return run_pipeline(ctx, frame).to_dict()


def route_session_update(body: dict) -> dict:
    sid    = body.get("session_id", str(uuid.uuid4()))
    frames = body.get("frames", [])
    ctx    = registry.get_or_create(sid)
    done   = 0
    last   = None
    for fd in frames:
        fd["session_id"] = sid
        try:
            last = run_pipeline(ctx, PerceptionFrame.from_dict(fd))
            done += 1
        except Exception:
            pass
    if not last:
        return {"error": "No frames processed", "frames_processed": 0}
    return {"session_id": sid, "frames_processed": done,
            "latest_analysis": last.to_dict()}


def route_report(sid: str) -> dict:
    ctx = registry.get(sid)
    if not ctx:
        return {"error": f"Session '{sid}' not found", "_status": 404}
    return build_report(ctx).to_dict()


def route_insight(sid: str) -> dict:
    import numpy as np
    from schemas import CognitiveState

    ctx = registry.get(sid)
    if not ctx:
        return {"error": "Session not found", "_status": 404}
    if not ctx.focus_hist:
        return {"insight": "Keep camera active — collecting data...",
                "session_id": sid, "current_state": "idle",
                "score": 0.0, "grade": "—", "trend": "stable"}

    fa = np.array(list(ctx.focus_hist))
    sa = np.array(list(ctx.stress_hist))
    la = np.array(list(ctx.load_hist))

    mock = CognitiveState(
        focus_score    = round(float(fa.mean()), 1),
        cognitive_load = round(float(la.mean()), 1),
        stress_level   = round(float(sa.mean()), 1),
        fatigue_level  = 20.0,
        engagement     = round(float(fa.mean()) * 0.85, 1),
        confidence     = 0.88,
        dominant_state = ctx.last_state if ctx.last_state != "idle" else "distracted",
    )
    tf         = ctx.feature_module.extract(30)
    long_trend = ctx.feature_module.extract_long_trend()

    if not ctx.buffer.is_ready(1):
        return {"insight": "Initializing...", "session_id": sid}

    dummy  = ctx.buffer.last_n(1)[0]
    re     = ReasoningEngine()
    bs     = BehavioralScorer()
    expl   = re.explain(dummy, tf, mock, long_trend)
    pred   = ctx.predictor.predict(mock.dominant_state, tf)
    score  = bs.score(mock, tf)
    text   = insight_gen.generate(mock, expl, pred, score, long_trend)

    return {
        "insight": text, "session_id": sid,
        "current_state": ctx.last_state,
        "score": score.overall, "grade": score.grade, "label": score.label,
        "trend": long_trend.get("direction", "stable"),
        "recommendation": expl.recommendation,
        "contributing_signals": expl.contributing_signals,
        "trend_note": expl.trend_note,
    }


def route_delete(sid: str) -> dict:
    if sid in registry._store:
        del registry._store[sid]
        return {"deleted": True, "session_id": sid}
    return {"error": "Session not found", "_status": 404}


def route_stats() -> dict:
    return {"active_sessions": registry.active_count(), "timestamp": time.time()}


# ════════════════════════════════════════════════════════════════
# HTTP HANDLER
# ════════════════════════════════════════════════════════════════

def _index_bytes() -> bytes:
    path = Path(_HERE) / "index.html"
    return path.read_bytes()


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}]  {fmt % args}")

    # ── helpers ──────────────────────────────

    def _json(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode()) if n else {}

    def _dispatch(self, result: dict):
        status = result.pop("_status", 200) if "_status" in result else 200
        self._json(result, status)

    # ── verbs ─────────────────────────────────

    def do_OPTIONS(self):
        self._json({})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = _index_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        path = urlparse(self.path).path.rstrip("/")
        try:
            if   path == "/health":          self._json(route_health())
            elif path == "/sessions/stats":  self._json(route_stats())
            elif "/report"  in path:         self._dispatch(route_report(path.split("/")[2]))
            elif "/insight" in path:         self._dispatch(route_insight(path.split("/")[2]))
            else:                            self._json({"error": "Not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            body = self._body()
            if   path == "/analyze":         self._json(route_analyze(body))
            elif path == "/session/update":  self._json(route_session_update(body))
            else:                            self._json({"error": "Not found"}, 404)
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_DELETE(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            parts = path.split("/")
            if len(parts) >= 3:
                self._dispatch(route_delete(parts[2]))
            else:
                self._json({"error": "Not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


# ════════════════════════════════════════════════════════════════
# FASTAPI WRAPPER  (optional — used if uvicorn is installed)
# ════════════════════════════════════════════════════════════════

def _try_fastapi():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(title="CBAS v3", version="3.0.0")
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        from fastapi.responses import HTMLResponse

        @app.get("/", response_class=HTMLResponse)
        def index():
            return (Path(_HERE) / "index.html").read_text(encoding="utf-8")

        @app.get("/index.html", response_class=HTMLResponse)
        def index_html():
            return (Path(_HERE) / "index.html").read_text(encoding="utf-8")

        @app.get("/health")
        def health(): return route_health()

        @app.get("/sessions/stats")
        def stats(): return route_stats()

        @app.post("/analyze")
        def analyze(body: dict): return route_analyze(body)

        @app.post("/session/update")
        def update(body: dict): return route_session_update(body)

        @app.get("/session/{sid}/report")
        def report(sid: str): return route_report(sid)

        @app.get("/session/{sid}/insight")
        def insight(sid: str): return route_insight(sid)

        @app.delete("/session/{sid}")
        def delete(sid: str): return route_delete(sid)

        return app
    except ImportError:
        return None


# Expose for uvicorn: uvicorn server:app
app = _try_fastapi()


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="CBAS v2 API Server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()

    # Try uvicorn + FastAPI first
    try:
        import uvicorn
        if app:
            print(f"\n  CBAS v2  —  FastAPI + Uvicorn")
            print(f"  http://{args.host}:{args.port}\n")
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
            return
    except ImportError:
        pass

    # Stdlib fallback (zero external deps)
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"\n  CBAS v2  —  Native HTTP Server  (no extra packages needed)")
    print(f"  Listening on  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        srv.server_close()


if __name__ == "__main__":
    main()
