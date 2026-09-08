from __future__ import annotations

import time
import uuid
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.features import (
    face_features,
    gaze_features,
    heuristic_load_score,
    performance_features,
    pose_features,
)

st.set_page_config(
    page_title="CognitiveSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 750; margin-bottom: 0.1rem;}
    .subtitle {color: #5c677d; margin-bottom: 1rem;}
    .badge {display:inline-block; padding:0.25rem 0.55rem; border-radius:999px; background:#eef2ff; font-size:0.8rem; margin-right:0.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 CognitiveSense AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-time multimodal behavioral-signal research platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">Computer Vision</span><span class="badge">Gaze</span>'
    '<span class="badge">Posture</span><span class="badge">Human–AI Research</span>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Research session")
    participant_id = st.text_input("Participant ID", "P001")
    session_id = st.text_input("Session ID", f"S-{uuid.uuid4().hex[:6].upper()}")
    task_name = st.selectbox("Task", ["Baseline", "Mental arithmetic", "2-back", "Stroop-like response"])
    difficulty = st.select_slider("Difficulty", options=[1, 2, 3, 4, 5], value=3)
    st.markdown("---")
    st.subheader("Modalities")
    enable_face = st.checkbox("Facial behavior", True)
    enable_gaze = st.checkbox("Gaze proxy", True)
    enable_pose = st.checkbox("Posture", True)
    st.markdown("---")
    st.warning(
        "Research prototype only. The live score is a heuristic proxy until a validated model is trained on labeled participant data."
    )

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=600)
if "task_rows" not in st.session_state:
    st.session_state.task_rows = []
if "task_started" not in st.session_state:
    st.session_state.task_started = False
if "task_start_time" not in st.session_state:
    st.session_state.task_start_time = None
if "last_features" not in st.session_state:
    st.session_state.last_features = {}

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def process_frame(img: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb) if (enable_face or enable_gaze) else None
    pose_result = pose.process(rgb) if enable_pose else None

    features: dict[str, float] = {}
    if enable_face and face_result and face_result.multi_face_landmarks:
        features.update(face_features(face_result.multi_face_landmarks[0].landmark))
    if enable_gaze and face_result and face_result.multi_face_landmarks:
        features.update(gaze_features(face_result.multi_face_landmarks[0].landmark))
    if enable_pose and pose_result and pose_result.pose_landmarks:
        features.update(pose_features(pose_result.pose_landmarks.landmark))

    # Performance values are supplied by the interactive task. During baseline/demo mode
    # these remain neutral rather than pretending a task measurement occurred.
    features.update(performance_features(1, 1000.0, 0.0))
    score = heuristic_load_score(features)

    overlay = img.copy()
    cv2.rectangle(overlay, (12, 12), (420, 68), (20, 25, 35), -1)
    cv2.putText(
        overlay,
        f"Cognitive-load proxy: {score:0.0f}/100",
        (25, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )
    return overlay, {**features, "proxy_score": float(score)}


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    output, features = process_frame(img)
    now = time.time()
    row = {
        "participant_id": participant_id,
        "session_id": session_id,
        "timestamp": now,
        "task": task_name,
        "difficulty": difficulty,
        **features,
    }
    st.session_state.last_features = features
    st.session_state.history.append(row)
    return av.VideoFrame.from_ndarray(output, format="bgr24")


tab_live, tab_task, tab_data, tab_research = st.tabs(
    ["Live monitor", "Cognitive task", "Research data", "Research protocol"]
)

with tab_live:
    st.subheader("Live multimodal monitor")
    st.caption("The video overlay is the real-time component. Enable only the modalities you intend to study.")
    webrtc_streamer(
        key="cognitive-load-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    cols = st.columns(4)
    latest = st.session_state.last_features
    score = latest.get("proxy_score")
    cols[0].metric("Load proxy", "—" if score is None else f"{score:.0f}/100")
    cols[1].metric("Face", "ON" if enable_face else "OFF")
    cols[2].metric("Gaze", "ON" if enable_gaze else "OFF")
    cols[3].metric("Posture", "ON" if enable_pose else "OFF")

    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        hist["elapsed_s"] = hist["timestamp"] - hist["timestamp"].iloc[0]
        st.line_chart(hist.set_index("elapsed_s")["proxy_score"])

with tab_task:
    st.subheader("Built-in behavioral task")
    st.write("Use this task to create measured reaction-time and accuracy features instead of placeholder performance values.")

    c1, c2, c3 = st.columns(3)
    if c1.button("Start 10-trial task", use_container_width=True):
        st.session_state.task_started = True
        st.session_state.task_start_time = time.perf_counter()
        st.session_state.task_rows = []

    if st.session_state.task_started:
        trial_no = len(st.session_state.task_rows) + 1
        target = int((trial_no * 7 + difficulty * 3) % 9)
        st.markdown(f"### Trial {trial_no}/10")
        st.write(f"Target value: **{target}**")
        answer = st.radio("Select your answer", list(range(9)), horizontal=True, key=f"answer_{trial_no}")
        if st.button("Record response", key=f"record_{trial_no}"):
            reaction_ms = (time.perf_counter() - st.session_state.task_start_time) * 1000
            correct = int(int(answer) == target)
            st.session_state.task_rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "task": task_name,
                    "difficulty": difficulty,
                    "trial": trial_no,
                    "target": target,
                    "answer": int(answer),
                    "correct": correct,
                    "reaction_time_ms": reaction_ms,
                    "timestamp": time.time(),
                }
            )
            st.session_state.task_start_time = time.perf_counter()
            if len(st.session_state.task_rows) >= 10:
                st.session_state.task_started = False
                st.success("Task complete. Export the trial data from the Research data tab.")
            st.rerun()

    if st.session_state.task_rows:
        task_df = pd.DataFrame(st.session_state.task_rows)
        st.dataframe(task_df, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Research dataset")
    feature_df = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()
    task_df = pd.DataFrame(st.session_state.task_rows) if st.session_state.task_rows else pd.DataFrame()

    st.write(f"Feature windows: **{len(feature_df)}** | Task trials: **{len(task_df)}**")
    if not feature_df.empty:
        st.dataframe(feature_df.tail(25), use_container_width=True, hide_index=True)
        st.download_button(
            "Download multimodal feature log",
            feature_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{participant_id}_{session_id}_features.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not task_df.empty:
        st.download_button(
            "Download task-performance log",
            task_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{participant_id}_{session_id}_task.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if st.button("Clear current session", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.task_rows = []
        st.session_state.last_features = {}
        st.rerun()

with tab_research:
    st.subheader("Research protocol")
    st.markdown(
        """
        **Primary research question**  
        Does combining behavioral signals improve cognitive-load estimation for participants not seen during training?

        **Core comparison**  
        Face only → Gaze only → Posture only → pairwise fusion → all modalities.

        **Evaluation rule**  
        Split by participant, not by random frame, so test data contains people excluded from model training.

        **Minimum label design**  
        Pair controlled task difficulty with subjective workload and objective task performance. Do not treat the webcam proxy itself as ground truth.

        **Recommended reporting**  
        Accuracy, macro-F1, ROC-AUC where applicable, confidence intervals, participant counts, and modality ablations.
        """
    )
    st.info("This page is the experiment plan; the trained-model stage should use collected participant data and a preregistered evaluation protocol.")
