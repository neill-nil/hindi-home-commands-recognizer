"""
app.py — Hindi Smart-Home Command Recognizer (T11.3)
UI redesigned to match reference design: lavender bg, white cards, magenta mic button.
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
    box-sizing: border-box;
}

/* ── Lavender gradient background ── */
.stApp {
    background: linear-gradient(160deg, #e8e0f7 0%, #f3effe 40%, #ede8fb 70%, #ddd6f3 100%) !important;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 680px !important;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 500;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(180,160,230,0.35);
    color: #5b21b6;
    letter-spacing: 0.01em;
    margin-bottom: 10px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e;
    animation: blink 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Big heading ── */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin: 0 0 10px 0;
}
.main-subtitle {
    font-size: 15px;
    color: #6b7280;
    font-weight: 400;
    margin: 0 0 28px 0;
    line-height: 1.5;
}

/* ── White card ── */
.ui-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 28px 28px;
    margin-bottom: 16px;
    box-shadow: 0 2px 20px rgba(120,90,200,0.08);
    border: 1px solid rgba(180,160,230,0.18);
}

.card-title {
    font-size: 17px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 4px 0;
}
.card-subtitle {
    font-size: 13px;
    color: #9ca3af;
    margin: 0 0 24px 0;
}

/* ── Mic wrapper (labels below the button) ── */
.mic-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 8px 0 4px;
}
.mic-label {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.2px;
    font-family: monospace;
    margin-top: 4px;
}
.mic-hint {
    font-size: 12.5px;
    color: #7c3aed;
    margin-top: -6px;
}

/* ── Mic button: circular, centred via columns in Python ── */
div[data-testid="stButton"] > button {
    width: 120px !important;
    height: 120px !important;
    border-radius: 50% !important;
    border: none !important;
    padding: 0 !important;
    /* Layer SVG icon ON TOP of gradient — fixes override bug */
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm-1 18.93V22h2v-2.07A8.001 8.001 0 0 0 20 12h-2a6 6 0 0 1-12 0H4a8.001 8.001 0 0 0 7 7.93z'/%3E%3C/svg%3E"),
        linear-gradient(135deg, #d500f9 0%, #a21caf 100%) !important;
    background-size: 44px 44px, 100% 100% !important;
    background-position: center, center !important;
    background-repeat: no-repeat, no-repeat !important;
    color: transparent !important;
    font-size: 0 !important;
    box-shadow: 0 20px 60px -10px rgba(192,38,211,0.55) !important;
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    transform: scale(1.07) !important;
    box-shadow: 0 24px 64px -8px rgba(192,38,211,0.65) !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.94) !important;
    box-shadow: 0 8px 24px rgba(192,38,211,0.4) !important;
}

/* Pulse ring on hold */
@keyframes mic-pulse {
    0%   { box-shadow: 0 0 0 0   rgba(192,38,211,0.6),  0 20px 60px -10px rgba(192,38,211,0.55); }
    70%  { box-shadow: 0 0 0 26px rgba(192,38,211,0.0),  0 20px 60px -10px rgba(192,38,211,0.55); }
    100% { box-shadow: 0 0 0 0   rgba(192,38,211,0.0),  0 20px 60px -10px rgba(192,38,211,0.55); }
}
div[data-testid="stButton"] > button.mic-held {
    animation: mic-pulse 0.85s cubic-bezier(0.215,0.61,0.355,1) infinite !important;
}

/* Waveform bars (shown via JS class) */
@keyframes wave {
    0%, 100% { transform: scaleY(0.3); }
    50%       { transform: scaleY(1); }
}
.wave-bars { display:flex; align-items:flex-end; justify-content:center; gap:3px; height:28px; margin-top:4px; }
.wave-bar  { width:4px; border-radius:2px; background:linear-gradient(135deg,#d500f9,#a21caf); transform-origin:bottom; }

/* ── Upload zone ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(140,100,220,0.35) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    background: rgba(245,240,255,0.5) !important;
    text-align: center !important;
}
div[data-testid="stFileUploader"] label {
    color: #6b21a8 !important;
    font-weight: 500 !important;
}
.upload-hint {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 6px;
    text-align: center;
}
.upload-icon {
    font-size: 28px;
    color: #7c3aed;
    display: block;
    text-align: center;
    margin-bottom: 6px;
}

/* ── Result card (predicted command) ── */
.result-banner {
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #fdf4ff, #f5f3ff);
    border: 1px solid rgba(180,120,230,0.25);
}
.result-banner-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #a855f7;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
}
.result-banner-source {
    font-size: 10.5px;
    font-weight: 500;
    color: #9ca3af;
    letter-spacing: 0.5px;
}
.result-main-row {
    display: flex;
    align-items: center;
    gap: 16px;
}
.result-icon-box {
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
}
.result-text-block { flex: 1; }
.result-hindi {
    font-size: 22px;
    font-weight: 700;
    color: #1a1a2e;
    font-family: 'Noto Sans Devanagari', sans-serif;
    line-height: 1.2;
}
.result-romanized {
    font-size: 12.5px;
    color: #6b7280;
    margin-top: 2px;
}
.result-conf-block {
    text-align: right;
    flex-shrink: 0;
}
.result-conf-num {
    font-size: 28px;
    font-weight: 800;
    color: #7c3aed;
    line-height: 1;
}
.result-conf-label {
    font-size: 9px;
    letter-spacing: 1px;
    color: #9ca3af;
    text-transform: uppercase;
    margin-top: 2px;
}
.result-progress {
    margin-top: 14px;
    height: 5px;
    border-radius: 3px;
    background: rgba(180,120,230,0.15);
    overflow: hidden;
}
.result-progress-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #a855f7, #c026d3);
    transition: width 0.6s ease;
}

/* ── Command list ── */
.cmd-list-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 16px;
}
.cmd-list-title {
    font-size: 17px;
    font-weight: 700;
    color: #111827;
}
.cmd-list-count {
    font-size: 13px;
    color: #9ca3af;
}
.cmd-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border-radius: 14px;
    background: rgba(248,245,255,0.7);
    border: 1px solid rgba(180,160,230,0.15);
    margin-bottom: 8px;
    transition: background 0.15s;
}
.cmd-row:hover { background: rgba(240,235,255,0.9); }
.cmd-row-icon {
    width: 40px; height: 40px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
    background: rgba(240,235,255,0.8);
}
.cmd-row-body { flex: 1; min-width: 0; }
.cmd-row-hindi {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a2e;
    font-family: 'Noto Sans Devanagari', sans-serif;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cmd-row-num {
    font-size: 11px;
    color: #9ca3af;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}
.cmd-row-roman {
    font-size: 11.5px;
    color: #9ca3af;
    margin-top: 2px;
}
.cmd-row-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(220,200,255,0.5);
    color: #7c3aed;
    flex-shrink: 0;
}

/* ── Divider ── */
.divider { height: 1px; background: rgba(180,150,230,0.18); margin: 20px 0; }

/* ── Transcript ── */
.transcript-box {
    background: rgba(245,240,255,0.7);
    border-left: 3px solid #a855f7;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 14px;
    color: #374151;
    margin-top: 12px;
    font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
}

/* ── Status bar ── */
.status-bar {
    text-align: center; padding: 12px;
    border-radius: 10px; font-weight: 500; font-size: 14px;
    margin: 8px 0;
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(180,150,230,0.3);
    color: #5b21b6;
}
.status-recording { color: #ef4444 !important; border-color: rgba(239,68,68,0.3) !important; }

/* ── Audio player ── */
audio { border-radius: 10px !important; width: 100% !important; margin-top: 8px; }

/* ── Info overrides ── */
.stAlert { border-radius: 10px !important; }

/* ── Sparkle icon in card header ── */
.card-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
}
.card-header-icon {
    font-size: 22px;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
MAX_DURATION = 4
CHANNELS     = 1

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("last_result", None),
    ("last_transcript", ""),
    ("last_audio_bytes", None),
    ("last_source", "microphone"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Model detection ───────────────────────────────────────────────────────────
USE_WHISPER_FT = load_whisper_model()
USE_ML         = (not USE_WHISPER_FT) and load_models()

# ── Extra metadata for commands ───────────────────────────────────────────────
COMMAND_META = {
    "light_on":  {"hindi": "बत्ती चालू करो", "roman": "batti chaalu karo", "action": "Turn on the light",  "category": "LIGHTS", "icon_bg": "#fef9c3", "emoji": "💡"},
    "light_off": {"hindi": "बत्ती बंद करो",  "roman": "batti band karo",   "action": "Turn off the light", "category": "LIGHTS", "icon_bg": "#e0f2fe", "emoji": "🔵"},
    "fan_on":    {"hindi": "पंखा चालू करो","roman": "pankha chaalu karo","action":"Turn on the fan",  "category": "FAN",    "icon_bg": "#e0f2fe", "emoji": "🌀"},
    "fan_off":   {"hindi": "पंखा बंद करो", "roman": "pankha band karo","action": "Turn off the fan",  "category": "FAN",    "icon_bg": "#fef3c7", "emoji": "🟧"},
    "ac_on":     {"hindi": "एसी चालू करो", "roman": "AC chaalu karo",  "action": "Turn on the AC",    "category": "AC",     "icon_bg": "#dbeafe", "emoji": "❄️"},
    "ac_off":    {"hindi": "एसी बंद करो",  "roman": "AC band karo",    "action": "Turn off the AC",   "category": "AC",     "icon_bg": "#fee2e2", "emoji": "🌡️"},
    "tv_on":     {"hindi": "टीवी चालू करो","roman": "TV chaalu karo",  "action": "Turn on the TV",    "category": "TV",     "icon_bg": "#dbeafe", "emoji": "📺"},
    "tv_off":    {"hindi": "टीवी बंद करो", "roman": "TV band karo",    "action": "Turn off the TV",   "category": "TV",     "icon_bg": "#fee2e2", "emoji": "🚫"},
    "music_on":  {"hindi": "गाना बजाओ",    "roman": "gaana bajao",     "action": "Play music",         "category": "MUSIC",  "icon_bg": "#f3e8ff", "emoji": "🎵"},
    "music_off": {"hindi": "गाना रोको",    "roman": "gaana roko",      "action": "Stop music",         "category": "MUSIC",  "icon_bg": "#fce7f3", "emoji": "⏸️"},
}

# ── RESULT DISPLAY HELPER ─────────────────────────────────────────────────────
def show_result(result: dict, source: str = "microphone"):
    cmd    = result.get("command", "unknown")
    conf   = result.get("confidence", 0.0)
    transcript = result.get("transcript", "")
    meta   = COMMAND_META.get(cmd)

    if meta:
        icon_bg  = meta["icon_bg"]
        emoji    = meta["emoji"]
        hindi    = meta["hindi"]
        roman    = meta["roman"]
        action   = meta["action"]
        conf_pct = conf * 100
    else:
        icon_bg, emoji = "#fee2e2", "❓"
        hindi, roman, action = "Unknown", "unknown", "Command not recognised"
        conf_pct = 0.0

    st.markdown(f"""
    <div class="result-banner">
        <div class="result-banner-label">
            <span>PREDICTED COMMAND</span>
            <span class="result-banner-source">via {source}</span>
        </div>
        <div class="result-main-row">
            <div class="result-icon-box" style="background:{icon_bg};">{emoji}</div>
            <div class="result-text-block">
                <div class="result-hindi">{hindi}</div>
                <div class="result-romanized">{roman} · {action}</div>
            </div>
            <div class="result-conf-block">
                <div class="result-conf-num">{conf_pct:.1f}%</div>
                <div class="result-conf-label">CONFIDENCE</div>
            </div>
        </div>
        <div class="result-progress">
            <div class="result-progress-fill" style="width:{min(conf_pct,100):.1f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if transcript and not USE_ML:
        st.markdown(
            f"<div class='transcript-box'>🎙 <em>{transcript}</em></div>",
            unsafe_allow_html=True,
        )


# ── HEADER ─────────────────────────────────────────────────────────────────────
if USE_WHISPER_FT:
    pill_text = "Whisper Fine-Tuned Classifier · Active"
elif USE_ML:
    pill_text = "MFCC + SVM Classifier · Active"
else:
    pill_text = "Whisper-base + String Match · Active"

st.markdown(f"""
<div class="status-pill">
    <span class="status-dot"></span>
    {pill_text}
</div>
<div class="main-title">Hindi Smart-Home</div>
<div class="main-subtitle">Voice command recognizer · trained on 10 spoken Hindi commands</div>
""", unsafe_allow_html=True)


# ── SHOW CACHED RESULT (above input cards) ────────────────────────────────────
result_area = st.empty()
if st.session_state.last_result:
    with result_area.container():
        show_result(st.session_state.last_result, st.session_state.last_source)
        if st.session_state.last_audio_bytes and st.session_state.last_source == "microphone":
            st.audio(st.session_state.last_audio_bytes, format="audio/wav")


# ── MIC CARD ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ui-card" style="padding-bottom:32px;">
    <div class="card-header-row">
        <div>
            <div class="card-title">Speak a command</div>
            <div class="card-subtitle">Press &amp; hold the mic, then release</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

status_ph = st.empty()

def run_prediction(audio_flat: np.ndarray) -> dict:
    if USE_WHISPER_FT:
        status_ph.markdown("<div class='status-bar'>Classifying with fine-tuned Whisper…</div>", unsafe_allow_html=True)
        return predict_whisper(audio_flat, SAMPLE_RATE)
    elif USE_ML:
        status_ph.markdown("<div class='status-bar'>Classifying with MFCC SVM…</div>", unsafe_allow_html=True)
        return predict_audio_direct(audio_flat, SAMPLE_RATE)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio_flat, SAMPLE_RATE)
        status_ph.markdown("<div class='status-bar'>Transcribing…</div>", unsafe_allow_html=True)
        try:
            transcript = transcribe(tmp_path)
        finally:
            try: os.unlink(tmp_path)
            except: pass
        result = match_command(transcript)
        result["transcript"] = transcript
        return result


# ── Mic button: centred with columns ──────────────────────────────────────────
if not SOUNDDEVICE_AVAILABLE:
    st.warning("Mic unavailable — run `sudo apt install portaudio19-dev` then restart.\n\nUse the Upload section below.")
else:
    _l, _mid, _r = st.columns([3, 2, 3])
    with _mid:
        clicked = st.button("mic", key="mic_btn")
        # Labels inside same column — perfectly aligned with button above
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; gap:5px; padding-top:4px;">
            <div class="mic-label" id="mic-label-text">Hold to speak</div>
            <div class="mic-hint"  id="mic-hint-text">Press &amp; hold the mic</div>
        </div>
        """, unsafe_allow_html=True)

    # Wave bars + JS — centred via flex; JS IDs reference the labels now inside the column
    st.markdown("""
    <div style="display:flex; justify-content:center; margin-top:-4px; padding-bottom:8px;">
        <div class="wave-bars" id="mic-wave" style="display:none;">
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.00s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.07s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.14s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.21s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.28s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.35s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.42s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.49s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.56s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.63s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.70s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.77s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.84s infinite;"></span>
          <span class="wave-bar" style="animation:wave 0.9s ease-in-out 0.91s infinite;"></span>
        </div>
    </div>
    <script>
    (function() {
        function attach() {
            var btn = document.querySelector('[data-testid="stButton"] > button');
            var lbl = document.getElementById('mic-label-text');
            var hnt = document.getElementById('mic-hint-text');
            var wav = document.getElementById('mic-wave');
            if (!btn || !lbl) { setTimeout(attach, 300); return; }
            var t0, raf;
            function tick() {
                var s = ((Date.now() - t0) / 1000).toFixed(1);
                lbl.textContent = s + 's';
                raf = requestAnimationFrame(tick);
            }
            btn.addEventListener('mousedown', function() {
                btn.classList.add('mic-held');
                t0 = Date.now(); tick();
                hnt.textContent = 'Release to stop';
                if (wav) wav.style.display = 'flex';
            });
            btn.addEventListener('touchstart', function() {
                btn.classList.add('mic-held');
                t0 = Date.now(); tick();
                hnt.textContent = 'Release to stop';
                if (wav) wav.style.display = 'flex';
            }, {passive:true});
            function stopAnim() {
                btn.classList.remove('mic-held');
                cancelAnimationFrame(raf);
                lbl.textContent = 'Hold to speak';
                hnt.textContent = 'Press & hold the mic';
                if (wav) wav.style.display = 'none';
            }
            btn.addEventListener('mouseup',    stopAnim);
            btn.addEventListener('touchend',   stopAnim);
            btn.addEventListener('mouseleave', stopAnim);
        }
        attach();
    })();
    </script>
    """, unsafe_allow_html=True)

    if clicked:
        result_area.empty()
        status_ph.markdown("<div class='status-bar status-recording'>🔴 Recording — speak now…</div>", unsafe_allow_html=True)
        audio = sd.rec(int(MAX_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=CHANNELS, dtype="float32")
        sd.wait()
        audio_flat = audio.flatten()
        buf = io.BytesIO()
        sf.write(buf, audio_flat, SAMPLE_RATE, format="WAV")
        st.session_state.last_audio_bytes = buf.getvalue()
        result = run_prediction(audio_flat)
        st.session_state.last_result     = result
        st.session_state.last_transcript = result.get("transcript", "")
        st.session_state.last_source     = "microphone"
        status_ph.empty()
        st.rerun()




# ── UPLOAD CARD ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ui-card" style="margin-top:0;">
    <div class="card-title">Upload audio file</div>
    <div class="card-subtitle">Test with any WAV · MP3 · M4A · FLAC file</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop audio file or click to upload",
    type=["wav", "mp3", "m4a", "flac"],
    label_visibility="visible",
)
st.markdown("<div class='upload-hint'>WAV · MP3 · M4A · FLAC — up to 200MB</div>", unsafe_allow_html=True)

if uploaded:
    raw_bytes = uploaded.read()
    st.audio(raw_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    with st.spinner("Analysing audio…"):
        if USE_WHISPER_FT:
            result = predict_whisper_from_file(tmp_path)
        elif USE_ML:
            result = predict_from_file_direct(tmp_path)
        else:
            transcript = transcribe(tmp_path)
            result = match_command(transcript)
            result["transcript"] = transcript

    try: os.unlink(tmp_path)
    except: pass

    st.session_state.last_result     = result
    st.session_state.last_transcript = result.get("transcript", "")
    st.session_state.last_source     = "file upload"
    st.session_state.last_audio_bytes = None
    show_result(result, source="file upload")


# ── COMMAND LIST ──────────────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="ui-card">
    <div class="cmd-list-header">
        <span class="cmd-list-title">Available commands</span>
        <span class="cmd-list-count">{len(COMMAND_META)} total</span>
    </div>
""", unsafe_allow_html=True)

rows_html = ""
for idx, (cmd_key, meta) in enumerate(COMMAND_META.items(), start=1):
    num_str = f"{idx:02d}"
    rows_html += f"""
    <div class="cmd-row">
        <div class="cmd-row-icon" style="background:{meta['icon_bg']};">{meta['emoji']}</div>
        <div class="cmd-row-body">
            <div class="cmd-row-hindi">{meta['hindi']} <span class="cmd-row-num">{num_str}</span></div>
            <div class="cmd-row-roman">{meta['roman']} &nbsp; {meta['action']}</div>
        </div>
        <div class="cmd-row-tag">{meta['category']}</div>
    </div>
    """

st.markdown(rows_html + "</div>", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
model_tag = "Whisper Fine-Tuned" if USE_WHISPER_FT else ("MFCC + SVM" if USE_ML else "Whisper-base + String Match")
st.markdown(
    f"<p style='text-align:center; color:#9ca3af; font-size:11px; margin-top:24px;'>"
    f"SMAI Assignment 3 &nbsp;·&nbsp; T11.3 &nbsp;·&nbsp; {model_tag}"
    f"</p>",
    unsafe_allow_html=True,
)
