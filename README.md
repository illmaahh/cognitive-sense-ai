# CognitiveSense AI v4

**Multimodal Behavioral Signals for Generalizable Cognitive Load Estimation**

CognitiveSense AI is a research-oriented, real-time browser system that combines **face, eye/gaze-proxy, hand, posture, audio-level, and task-performance signals** for cognitive-load research. It provides a live perception console, a controlled task/data-collection workflow, an interpretable baseline pipeline, and participant-independent evaluation utilities.

> **Research status:** the live cognitive-load score in the demo is a heuristic proxy. It is not a clinically validated measurement and must not be presented as a diagnosis of stress, depression, fatigue disorders, or any mental-health condition.

## What is live?

The `live_cbas/` console runs in the browser and can process camera frames continuously:

- Face mesh and eye landmarks
- Gaze-proxy features from iris/eye geometry
- Blink/eye-openness cues
- Head movement and upper-body pose
- Two-hand landmarks and rule-based gesture labels
- Live microphone level and browser speech recognition where supported
- Observable facial landmark cues such as smile-like, surprise-like, tension/concern, and mouth-open signals
- Real-time session charts and optional backend inference

The system reports **observable cues**, not a person's true emotional state. In particular, it does not claim that webcam signals can diagnose depression or reliably establish that someone is sad or crying.

## Run locally: one-command live console

Requirements: Python 3.11 recommended.

### Windows

```bat
cd cognitive_load_ai\live_cbas
START.bat
```

Then open `http://localhost:8000`.

### macOS/Linux

```bash
cd cognitive_load_ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn live_cbas.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and allow camera + microphone permissions.

### Streamlit research dashboard

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app is useful for controlled task collection and research-data export. The `live_cbas/` app is the polished live-demo layer.

## Live demonstration

Primary deployment: **https://cognitive-sense-ai.onrender.com**

The public console is intended for demonstration and research prototyping. Camera and microphone processing occurs in the browser; backend inference and session analytics are served by FastAPI.


## Live console features

The v4 console adds a visual neural-HUD layer with face/eye, hand, and posture landmarks plus connection lines, a live sensor matrix, gesture labels, gaze-proxy tracking, audio-level monitoring, speech support where the browser provides it, backend health state, and session/report export.

Gesture and expression labels are observable cues derived from landmark geometry. They are not claims about a person's true emotions or mental-health status.

## Deploy publicly

### Render

This repository includes `render.yaml`.

1. Push the project to GitHub.
2. Create a new Render Web Service from the repository.
3. Render will use the included build/start configuration.
4. Open the generated **HTTPS** URL and allow camera/microphone access.

The FastAPI server serves both the API and `live_cbas/index.html` from the same origin, so the public demo does not depend on a hard-coded `localhost` backend.

### Docker

```bash
docker build -t cognitive-load-ai .
docker run --rm -p 8000:8000 cognitive-load-ai
```

Open `http://localhost:8000`.

## Project architecture

```text
Browser camera + microphone
        │
        ├── Face / eyes / gaze proxy
        ├── Hands / gesture cues
        ├── Pose / movement
        └── Audio level / speech support
                │
                ▼
        Live feature state
                │
        ┌───────┴─────────┐
        ▼                 ▼
  Local visual UI     FastAPI backend
                          │
                    temporal features
                          │
                    cognitive proxy
                          │
                    insight / report

Research path:
participant → controlled task → derived features → labels → grouped evaluation → modality ablation
```

## Research question

> Does combining behavioral signals improve cognitive-load estimation for participants not seen during model training?

### Planned ablation

1. Face only
2. Gaze only
3. Posture only
4. Face + gaze
5. Face + posture
6. Gaze + posture
7. All modalities

### Evaluation principle

Do not randomly split adjacent video windows when claiming user generalization. Use **participant-grouped evaluation** so test participants are absent from training.

## Research data design

For each time window, collect only what is necessary for the study:

- participant ID
- session ID
- task and difficulty
- derived facial/gaze/pose features
- objective task performance
- subjective workload measurement
- protocol-defined target label

Keep raw camera/audio media private unless participants have explicitly consented to storage and reuse. Feature-only research datasets can reduce privacy exposure.

## Quality and scientific boundaries

The project deliberately separates three layers:

1. **Perception:** what the sensors visibly/audio-wise observe.
2. **Proxy inference:** a demo-only heuristic for immediate feedback.
3. **Validated research model:** to be trained only after real participant data and a pre-specified labeling/evaluation protocol exist.

Do not claim that facial landmarks alone can reliably identify depression, mental illness, or a person's true internal emotional state. Any such classifier would require a separate, ethically designed validation study and appropriate clinical methodology.

## Testing

```bash
pytest -q
```

The repository includes unit tests for feature scoring and the participant-independent evaluation guardrails.

## GitHub checklist

Before publishing:

- Replace placeholder social links in the app/profile if present.
- Do not commit participant raw video/audio.
- Keep `.env`, local secrets, and generated caches out of Git.
- Add your own dataset only after consent and de-identification.
- Report measured results only after the experiment is complete.

## License

MIT. See `LICENSE`.
