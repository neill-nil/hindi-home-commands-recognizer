"""
app.py — Hindi Smart-Home Command Recognizer (T11.3)
Sleek dark UI with live audio playback and ML-first pipeline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import numpy as np
import streamlit as st
import soundfile as sf
import tempfile

import config
from src.transcribe import transcribe
from src.matcher import match_command
from src.audio_pipeline import predict_audio_direct, predict_from_file_direct, load_models
from src.whisper_pipeline import predict_whisper, predict_whisper_from_file, load_whisper_model

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

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
    box-sizing: border-box;
}

/* ── Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 20%, #1a1040 0%, #0d0d1a 50%, #0a0a0f 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 720px; }

/* ── Headings ── */
h1 { color: #fff !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h3 { color: #a78bfa !important; font-weight: 600 !important; font-size: 0.95rem !important;
     text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem !important; }

/* ── Status pill ── */
.pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 999px; font-size: 12px;
    font-weight: 600; letter-spacing: 0.5px;
}
.pill-ml   { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.pill-whisper { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.pill-dot  { width: 7px; height: 7px; border-radius: 50%; background: currentColor;
             animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Record button ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    padding: 18px 0 !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    border: none !important;
    background: linear-gradient(135deg, #7c3aed, #a855f7, #ec4899) !important;
    color: white !important;
    letter-spacing: 0.5px !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.55) !important;
    filter: brightness(1.08) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    filter: brightness(0.95) !important;
}

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin: 12px 0;
}

/* ── Result card ── */
.result-card {
    display: flex; align-items: center; gap: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px; padding: 22px 28px;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.result-card.success { border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.05); }
.result-card.error   { border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.05); }
.result-emoji { font-size: 52px; line-height: 1; flex-shrink: 0; }
.result-info  { flex: 1; }
.result-label { font-size: 26px; font-weight: 800; letter-spacing: -0.3px; }
.result-meta  { font-size: 12px; color: #6b7280; margin-top: 4px; font-family: 'Inter', monospace; }

/* ── Transcript box ── */
.transcript-box {
    background: rgba(0,0,0,0.35);
    border-left: 3px solid #7c3aed;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 17px;
    color: #d1d5db;
    margin: 8px 0;
    font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
    line-height: 1.5;
}

/* ── Status bar ── */
.status-bar {
    text-align: center; padding: 14px;
    border-radius: 12px; font-weight: 600; font-size: 15px;
    margin: 8px 0;
}
.status-recording { background: rgba(239,68,68,0.15); color: #f87171;
    border: 1px solid rgba(239,68,68,0.25); animation: pulse-border 1.2s ease-in-out infinite; }
.status-processing { background: rgba(124,58,237,0.15); color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.25); }
@keyframes pulse-border { 0%,100%{border-color:rgba(239,68,68,0.25)} 50%{border-color:rgba(239,68,68,0.7)} }

/* ── Command grid ── */
.cmd-tile {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 14px 8px; text-align: center;
    transition: all 0.2s ease; cursor: default;
}
.cmd-tile:hover { background: rgba(124,58,237,0.12); border-color: rgba(124,58,237,0.4); transform: translateY(-2px); }
.cmd-emoji { font-size: 26px; margin-bottom: 4px; }
.cmd-name  { font-size: 11px; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.8px; }
.cmd-hindi { font-size: 11px; color: #6b7280; margin-top: 2px; }

/* ── Divider ── */
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
           margin: 24px 0; }

/* ── Audio player dark override ── */
audio { filter: invert(0.85) hue-rotate(180deg) !important; border-radius: 8px !important; width: 100% !important; }

/* ── Info/warning overrides ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
MAX_DURATION = 4
CHANNELS     = 1

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("last_result", None),
    ("last_transcript", ""),
    ("last_audio_bytes", None),    # WAV bytes of last recording for playback
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Model detection (priority: Whisper fine-tuned > MFCC SVM > Whisper ASR) ──
USE_WHISPER_FT = load_whisper_model()   # fine-tuned Whisper classifier
USE_ML         = (not USE_WHISPER_FT) and load_models()  # MFCC SVM fallback

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏠 &nbsp;Hindi Smart-Home")
st.markdown("##### Voice Command Recognizer &nbsp;·&nbsp; T11.3")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Status pill
if USE_WHISPER_FT:
    st.markdown(
        "<div class='pill pill-ml'><span class='pill-dot'></span>Whisper Fine-Tuned Classifier Active</div>",
        unsafe_allow_html=True,
    )
elif USE_ML:
    st.markdown(
        "<div class='pill pill-ml'><span class='pill-dot'></span>MFCC + SVM Active (train Whisper for better accuracy)</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='pill pill-whisper'><span class='pill-dot'></span>Whisper-base + String Match (no model trained)</div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND GRID
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋  Available Commands", expanded=False):
    cols = st.columns(4)
    for i, (cmd, info) in enumerate(config.COMMAND_DISPLAY.items()):
        if cmd == "unknown":
            continue
        phrase = config.CANONICAL_PHRASES.get(cmd, "")
        with cols[i % 4]:
            st.markdown(
                f"<div class='cmd-tile'>"
                f"<div class='cmd-emoji'>{info['emoji']}</div>"
                f"<div class='cmd-name'>{info['label']}</div>"
                f"<div class='cmd-hindi'>{phrase}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MIC RECORDING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🎙️  Speak a Command")

status_ph    = st.empty()
audio_ph     = st.empty()    # playback player — cleared on new recording
result_ph    = st.empty()
transcript_ph = st.empty()


def run_prediction(audio_flat: np.ndarray) -> dict:
    """Priority: Whisper fine-tuned > MFCC SVM > Whisper ASR transcription."""
    if USE_WHISPER_FT:
        status_ph.markdown(
            "<div class='status-bar status-processing'>⚡ Classifying with fine-tuned Whisper …</div>",
            unsafe_allow_html=True,
        )
        return predict_whisper(audio_flat, SAMPLE_RATE)
    elif USE_ML:
        status_ph.markdown(
            "<div class='status-bar status-processing'>⚡ Classifying with MFCC SVM …</div>",
            unsafe_allow_html=True,
        )
        return predict_audio_direct(audio_flat, SAMPLE_RATE)
    else:
        # Save to temp for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio_flat, SAMPLE_RATE)
        status_ph.markdown(
            "<div class='status-bar status-processing'>🌀 Transcribing …</div>",
            unsafe_allow_html=True,
        )
        try:
            transcript = transcribe(tmp_path)
        finally:
            try: os.unlink(tmp_path)
            except: pass
        result = match_command(transcript)
        result["transcript"] = transcript
        return result


def record_and_predict():
    # Clear previous result and audio immediately
    audio_ph.empty()
    result_ph.empty()
    transcript_ph.empty()

    status_ph.markdown(
        "<div class='status-bar status-recording'>🔴 &nbsp;Recording — speak now!</div>",
        unsafe_allow_html=True,
    )

    audio = sd.rec(int(MAX_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()

    audio_flat = audio.flatten()

    # Save WAV bytes for playback
    buf = io.BytesIO()
    sf.write(buf, audio_flat, SAMPLE_RATE, format="WAV")
    st.session_state.last_audio_bytes = buf.getvalue()

    result = run_prediction(audio_flat)

    st.session_state.last_result     = result
    st.session_state.last_transcript = result.get("transcript", "")
    status_ph.empty()


# ── Record button ──────────────────────────────────────────────────────────────
if not SOUNDDEVICE_AVAILABLE:
    st.warning("🎤 **Mic unavailable** — run `sudo apt install portaudio19-dev` then restart.\n\nUse the Upload section below ↓")
elif st.button(f"🎙️  &nbsp; Record  ({MAX_DURATION}s) &nbsp; 🎙️"):
    record_and_predict()

# ── Show cached audio + result ─────────────────────────────────────────────────
if st.session_state.last_audio_bytes:
    audio_ph.audio(st.session_state.last_audio_bytes, format="audio/wav")

if st.session_state.last_result:
    res    = st.session_state.last_result
    cmd    = res["command"]
    info   = config.COMMAND_DISPLAY.get(cmd, config.COMMAND_DISPLAY["unknown"])
    conf   = res.get("confidence", 0.0)
    method = res.get("method", "")

    card_class = "success" if cmd != "unknown" else "error"
    color = info["color"]
    result_ph.markdown(
        f"<div class='result-card {card_class}'>"
        f"  <div class='result-emoji'>{info['emoji']}</div>"
        f"  <div class='result-info'>"
        f"    <div class='result-label' style='color:{color};'>{info['label']}</div>"
        f"    <div class='result-meta'>Confidence: {conf*100:.0f}% &nbsp;·&nbsp; Method: {method}</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.last_transcript and not USE_ML:
        transcript_ph.markdown(
            f"<div class='transcript-box'>🗣️ &nbsp; {st.session_state.last_transcript}</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📂  Upload Audio File")
st.caption("Test with any WAV · MP3 · M4A · FLAC file")

uploaded = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a", "flac"],
    label_visibility="collapsed",
)

if uploaded:
    raw_bytes = uploaded.read()

    # Always show player
    st.audio(raw_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    if USE_WHISPER_FT:
        with st.spinner("Classifying with fine-tuned Whisper …"):
            result = predict_whisper_from_file(tmp_path)
    elif USE_ML:
        with st.spinner("Classifying from audio …"):
            result = predict_from_file_direct(tmp_path)
    else:
        with st.spinner("Transcribing …"):
            transcript = transcribe(tmp_path)
        result = match_command(transcript)
        result["transcript"] = transcript

    try: os.unlink(tmp_path)
    except: pass

    cmd    = result["command"]
    info   = config.COMMAND_DISPLAY.get(cmd, config.COMMAND_DISPLAY["unknown"])
    conf   = result.get("confidence", 0.0)
    method = result.get("method", "")
    card_class = "success" if cmd != "unknown" else "error"
    color = info["color"]
    st.markdown(
        f"<div class='result-card {card_class}'>"
        f"  <div class='result-emoji'>{info['emoji']}</div>"
        f"  <div class='result-info'>"
        f"    <div class='result-label' style='color:{color};'>{info['label']}</div>"
        f"    <div class='result-meta'>Confidence: {conf*100:.0f}% &nbsp;·&nbsp; Method: {method}</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not USE_ML and result.get("transcript"):
        st.markdown(
            f"<div class='transcript-box'>🗣️ &nbsp; {result['transcript']}</div>",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
model_tag = "MFCC + SVM" if USE_ML else "Whisper-base + String Match"
st.markdown(
    f"<p style='text-align:center; color:#374151; font-size:11px;'>"
    f"SMAI Assignment 3 &nbsp;·&nbsp; T11.3 &nbsp;·&nbsp; {model_tag}"
    f"</p>",
    unsafe_allow_html=True,
)
