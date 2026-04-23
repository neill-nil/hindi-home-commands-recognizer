"""
app.py — Streamlit app for Hindi Smart-Home Command Recognizer (T11.3)

Features:
  - Hold-to-record button (press and hold, release to transcribe)
  - Real-time Whisper-tiny transcription
  - Command prediction via keyword + fuzzy matching
  - Simulated smart-home action (emoji-based feedback)

Run:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
import soundfile as sf
import tempfile

import config
from src.transcribe import transcribe
from src.matcher import match_command

# PortAudio / sounddevice is optional — needed only for mic recording
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except OSError:
    SOUNDDEVICE_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hindi Smart-Home Commander",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Command result card */
.result-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 24px 32px;
    margin: 16px 0;
    backdrop-filter: blur(10px);
    text-align: center;
}

.command-emoji { font-size: 72px; line-height: 1; margin-bottom: 8px; }
.command-label { font-size: 28px; font-weight: 700; letter-spacing: 1px; }
.command-sub   { font-size: 14px; color: #aaa; margin-top: 4px; }

/* Transcript box */
.transcript-box {
    background: rgba(0,0,0,0.3);
    border-left: 4px solid #6c63ff;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 18px;
    color: #e0e0e0;
    margin: 12px 0;
    font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
}

/* Record button styling */
.stButton > button {
    width: 100%;
    padding: 20px 0;
    font-size: 20px;
    font-weight: 700;
    border-radius: 50px;
    border: none;
    background: linear-gradient(90deg, #6c63ff, #e040fb);
    color: white;
    cursor: pointer;
    transition: all 0.2s ease;
    letter-spacing: 1px;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.5);
}
.stButton > button:active {
    transform: scale(0.98);
    background: linear-gradient(90deg, #e040fb, #6c63ff);
}

/* Section headers */
h1 { color: #ffffff; text-align: center; }
h3 { color: #c0b3ff; }

/* Status message */
.recording-indicator {
    text-align: center;
    color: #ff6b6b;
    font-size: 16px;
    font-weight: 600;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.4; }
    100% { opacity: 1; }
}

/* Unknown / error */
.unknown-card {
    background: rgba(255, 80, 80, 0.1);
    border: 1px solid rgba(255, 80, 80, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000
MAX_DURATION    = 5      # seconds max recording per press
CHANNELS        = 1

# ── Session state ─────────────────────────────────────────────────────────────
if "last_result"     not in st.session_state:
    st.session_state.last_result = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "status_msg"      not in st.session_state:
    st.session_state.status_msg = ""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏠 Hindi Smart-Home Commander")
st.markdown(
    "<p style='text-align:center; color:#aaa; margin-top:-12px;'>"
    "T11.3 · Whisper-tiny + String Match · Zero Training"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Command reference card ─────────────────────────────────────────────────────
with st.expander("📋 Available Commands (click to expand)"):
    cols = st.columns(4)
    for i, (cmd, info) in enumerate(config.COMMAND_DISPLAY.items()):
        if cmd == "unknown":
            continue
        phrase = config.CANONICAL_PHRASES.get(cmd, "")
        with cols[i % 4]:
            st.markdown(
                f"<div style='text-align:center; padding:8px;'>"
                f"<div style='font-size:28px'>{info['emoji']}</div>"
                f"<div style='font-size:12px; font-weight:600; color:#c0b3ff'>{info['label']}</div>"
                f"<div style='font-size:11px; color:#888;'>{phrase}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Recording section ─────────────────────────────────────────────────────────
st.markdown("### 🎙️ Record a Command")
st.markdown(
    "<p style='color:#aaa; font-size:14px;'>"
    "Click the button, speak your Hindi command clearly, then wait for the result."
    "</p>",
    unsafe_allow_html=True,
)

status_placeholder  = st.empty()
result_placeholder  = st.empty()
transcript_placeholder = st.empty()


def record_and_predict():
    """Record audio, transcribe with Whisper-tiny, predict command."""
    status_placeholder.markdown(
        "<div class='recording-indicator'>🔴 Recording … speak now!</div>",
        unsafe_allow_html=True,
    )

    audio = sd.rec(
        int(MAX_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()

    status_placeholder.markdown(
        "<div style='text-align:center; color:#6c63ff;'>⏳ Transcribing …</div>",
        unsafe_allow_html=True,
    )

    audio_flat = audio.flatten()

    # Save to temp file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_flat, SAMPLE_RATE)

    try:
        transcript = transcribe(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    result = match_command(transcript)
    result["transcript"] = transcript

    st.session_state.last_result     = result
    st.session_state.last_transcript = transcript
    status_placeholder.empty()

    return result


# ── Main record button ─────────────────────────────────────────────────────────
if not SOUNDDEVICE_AVAILABLE:
    st.warning(
        "🎤 **Mic recording unavailable** — PortAudio library not found.\n\n"
        "Fix: `sudo apt install portaudio19-dev` then restart the app.\n\n"
        "Meanwhile, use **Upload an Audio File** below to test ↓"
    )
elif st.button(f"🎙️  Hold to Record  ({MAX_DURATION}s)"):
    result = record_and_predict()

# ── Result display ─────────────────────────────────────────────────────────────
if st.session_state.last_result:
    res   = st.session_state.last_result
    cmd   = res["command"]
    info  = config.COMMAND_DISPLAY.get(cmd, config.COMMAND_DISPLAY["unknown"])
    conf  = res.get("confidence", 0.0)
    method = res.get("method", "")

    # Transcript box
    if st.session_state.last_transcript:
        transcript_placeholder.markdown(
            f"<div class='transcript-box'>"
            f"🗣️&nbsp;&nbsp;<em>{st.session_state.last_transcript}</em>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Result card
    extra_class = "unknown-card" if cmd == "unknown" else ""
    result_placeholder.markdown(
        f"<div class='result-card {extra_class}'>"
        f"<div class='command-emoji'>{info['emoji']}</div>"
        f"<div class='command-label' style='color:{info['color']};'>{info['label']}</div>"
        f"<div class='command-sub'>Confidence: {conf*100:.0f}%  ·  Method: {method}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── File upload fallback ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📂 Or Upload an Audio File")
uploaded = st.file_uploader(
    "Upload a WAV/MP3 file to test",
    type=["wav", "mp3", "m4a", "flac"],
    label_visibility="collapsed",
)

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    with st.spinner("Transcribing …"):
        transcript = transcribe(tmp_path)
    os.unlink(tmp_path)

    result = match_command(transcript)
    cmd    = result["command"]
    info   = config.COMMAND_DISPLAY.get(cmd, config.COMMAND_DISPLAY["unknown"])
    conf   = result.get("confidence", 0.0)

    st.markdown(
        f"<div class='transcript-box'>🗣️&nbsp;&nbsp;<em>{transcript}</em></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='result-card'>"
        f"<div class='command-emoji'>{info['emoji']}</div>"
        f"<div class='command-label' style='color:{info['color']};'>{info['label']}</div>"
        f"<div class='command-sub'>Confidence: {conf*100:.0f}%  ·  Method: {result['method']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:12px;'>"
    "SMAI Assignment 3 · T11.3 Hindi Smart-Home Commands · Whisper-tiny (39M, CPU)"
    "</p>",
    unsafe_allow_html=True,
)
