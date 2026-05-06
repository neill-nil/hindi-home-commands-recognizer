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
FLEURS_DIR = os.path.join(DATA_DIR, "fleurs_hindi")
MINED_LOCAL_FLEURS_DIR = os.path.join(DATA_DIR, "mined_local_fleurs")

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
}

# ── Keyword map for rule-based matching ───────────────────────────────────────
# Structure: {command_label: ([device_keywords], [state_keywords])}
# A command matches if ≥1 device keyword AND ≥1 state keyword appear in transcript.
KEYWORD_MAP = {
    "light_on":  (["लाइट", "light", "बत्ती", "दीपक", "रोशनी", "बल्ब", "चमक", "किरण"], ["चालू", "on", "जलाओ", "खोलो", "शुरू", "लगाओ", "चलाओ", "बढ़ाओ", "तेज़"]),
    "light_off": (["लाइट", "light", "बत्ती", "दीपक", "रोशनी", "बल्ब", "चमक", "किरण"], ["बंद", "off", "बुझाओ", "रोको", "खत्म", "कम", "बुझ"]),
    "fan_on":    (["पंखा", "fan", "पंखे", "हवा", "कूलर", "ब्लोअर"], ["चालू", "on", "चलाओ", "खोलो", "शुरू", "लगाओ", "बढ़ाओ", "तेज़", "स्टार्ट"]),
    "fan_off":   (["पंखा", "fan", "पंखे", "हवा", "कूलर", "ब्लोअर"], ["बंद", "off", "रोको", "खत्म", "कम", "रुक"]),
    "ac_on":     (["ac", "AC", "एसी", "एयर कंडीशनर", "ठंड", "ठंडा", "हीटर"], ["चालू", "on", "खोलो", "शुरू", "लगाओ", "चलाओ", "बढ़ाओ", "स्टार्ट"]),
    "ac_off":    (["ac", "AC", "एसी", "एयर कंडीशनर", "ठंड", "ठंडा", "हीटर"], ["बंद", "off", "रोको", "खत्म", "कम", "निकालो"]),
    "tv_on":     (["tv", "TV", "टीवी", "टेलीविज़न", "स्क्रीन", "चैनल", "फिल्म", "वीडियो", "शो"], ["चालू", "on", "खोलो", "शुरू", "लगाओ", "चलाओ", "दिखाओ", "स्टार्ट", "चले"]),
    "tv_off":    (["tv", "TV", "टीवी", "टेलीविज़न", "स्क्रीन", "चैनल", "फिल्म", "वीडियो", "शो"], ["बंद", "off", "रोको", "खत्म", "कम", "हटाओ"]),
}

# ── Matcher thresholds ─────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 60   # rapidfuzz score 0–100; below this → "unknown"

# ── UI display settings ────────────────────────────────────────────────────────
COMMAND_DISPLAY = {
    "light_on":  {"emoji": "💡", "label": "Light ON",  "color": "#f0c040"},
    "light_off": {"emoji": "🔦", "label": "Light OFF", "color": "#555555"},
    "fan_on":    {"emoji": "🌀", "label": "Fan ON",    "color": "#40a0f0"},
    "fan_off":   {"emoji": "🛑", "label": "Fan OFF",   "color": "#555555"},
    "ac_on":     {"emoji": "❄️",  "label": "AC ON",    "color": "#00cfff"},
    "ac_off":    {"emoji": "🔥", "label": "AC OFF",    "color": "#555555"},
    "tv_on":     {"emoji": "📺", "label": "TV ON",     "color": "#a060f0"},
    "tv_off":    {"emoji": "⏹️",  "label": "TV OFF",   "color": "#555555"},
    "unknown":   {"emoji": "❓", "label": "Unknown",   "color": "#ff4444"},
}
