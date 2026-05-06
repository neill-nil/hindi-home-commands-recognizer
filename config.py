"""
config.py — Central configuration for Hindi Smart-Home Command Recognizer (T11.3)

Edit COMMANDS and KEYWORD_MAP here to add/change commands.
"""

import os

# ── Project paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SELF_RECORDED_DIR = os.path.join(DATA_DIR, "self_recorded")
INDICVOICES_DIR = os.path.join(DATA_DIR, "indicvoices_hindi")

# ── Whisper settings ──────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"   # Options: tiny | base | small | medium | large
WHISPER_LANGUAGE = "hi"       # Force Hindi decoding

# ── 8 Smart-home commands (English slugs used as ground-truth labels) ─────────
COMMANDS = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "ac_on",
    "ac_off",
    "tv_on",
    "tv_off",
    "music_on",
    "music_off",
]

# ── Canonical Hindi phrases (what you'll say when recording) ──────────────────
# These are also used as the primary fuzzy-match reference strings.
CANONICAL_PHRASES = {
    "light_on":  "बत्ती चालू करो",
    "light_off": "बत्ती बंद करो",
    "fan_on":    "पंखा चालू करो",
    "fan_off":   "पंखा बंद करो",
    "ac_on":     "AC चालू करो",
    "ac_off":    "AC बंद करो",
    "tv_on":     "TV चालू करो",
    "tv_off":    "TV बंद करो",
    "music_on":  "गाना बजाओ",
    "music_off": "गाना रोको",
}

# ── Keyword map for rule-based matching ───────────────────────────────────────
# Structure: {command_label: ([device_keywords], [state_keywords])}
# A command matches if ≥1 device keyword AND ≥1 state keyword appear in transcript.
KEYWORD_MAP = {
    "light_on":  (["बत्ती", "batti", "लाइट", "light",  "दीपक"],        ["चालू", "on", "जलाओ", "खोलो", "शुरू"]),
    "light_off": (["बत्ती", "batti", "लाइट", "light",  "दीपक"],        ["बंद", "off", "बुझाओ"]),
    "fan_on":    (["पंखा", "fan", "पंखे"],                  ["चालू", "on", "चलाओ", "खोलो", "शुरू"]),
    "fan_off":   (["पंखा", "fan", "पंखे"],                  ["बंद", "off"]),
    "ac_on":     (["ac", "AC", "एसी", "एयर कंडीशनर"],       ["चालू", "on", "खोलो", "शुरू"]),
    "ac_off":    (["ac", "AC", "एसी", "एयर कंडीशनर"],       ["बंद", "off"]),
    "tv_on":     (["tv", "TV", "टीवी", "टेलीविज़न"],         ["चालू", "on", "खोलो", "शुरू"]),
    "tv_off":    (["tv", "TV", "टीवी", "टेलीविज़न"],         ["बंद", "off"]),
    "music_on":  (["गाना", "music", "song", "म्यूजिक", "गीत", "गाने"], ["बजाओ", "play", "चालू", "लगाओ", "शुरू", "चलाओ"]),
    "music_off": (["गाना", "music", "song", "म्यूजिक", "गीत", "गाने"], ["रोको", "stop", "बंद", "ठहरो"]),
}

# ── Matcher thresholds ─────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 60   # rapidfuzz score 0–100; below this → "unknown"

# ── UI display settings ────────────────────────────────────────────────────────
COMMAND_DISPLAY = {
    "light_on":  {"emoji": "💡", "label": "Light ON",  "color": "#64748b"},
    "light_off": {"emoji": "🔦", "label": "Light OFF", "color": "#94a3b8"},
    "fan_on":    {"emoji": "🌀", "label": "Fan ON",    "color": "#64748b"},
    "fan_off":   {"emoji": "🛑", "label": "Fan OFF",   "color": "#94a3b8"},
    "ac_on":     {"emoji": "❄️",  "label": "AC ON",    "color": "#64748b"},
    "ac_off":    {"emoji": "🔥", "label": "AC OFF",    "color": "#94a3b8"},
    "tv_on":     {"emoji": "📺", "label": "TV ON",     "color": "#64748b"},
    "tv_off":    {"emoji": "⏹️",  "label": "TV OFF",   "color": "#94a3b8"},
    "music_on":  {"emoji": "🎵", "label": "Music ON",  "color": "#64748b"},
    "music_off": {"emoji": "🔇", "label": "Music OFF", "color": "#94a3b8"},
    "unknown":   {"emoji": "❓", "label": "Unknown",   "color": "#ef4444"},
}
