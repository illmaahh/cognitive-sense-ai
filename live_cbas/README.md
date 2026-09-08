# Live Multimodal Console

This directory contains the real-time browser interface for CognitiveLoad AI.

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
uvicorn live_cbas.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

Windows shortcut:

```bat
cd live_cbas
START.bat
```

## What it detects

- Face and facial landmarks
- Eye openness, blink cues, and gaze proxy
- Head and upper-body movement
- Two hands and rule-based gesture categories
- Audio activity level
- Browser speech recognition where supported
- Observable facial landmark cues

## Backend

The same FastAPI process serves `/` and `/index.html` as well as `/health`, `/analyze`, and session endpoints. This makes local and HTTPS-hosted deployment simpler than using a fixed `localhost` API URL.

For split deployments, set the browser backend URL in local storage with the key `cbas_backend_url`.

## Safety / scientific boundary

The interface describes observable signals. It does not diagnose depression or infer a person's true emotional state. The cognitive-load score shown in the live demo is a heuristic proxy until a participant-trained model is validated.
