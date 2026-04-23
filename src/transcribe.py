"""
src/transcribe.py — Whisper-tiny transcription wrapper

Uses OpenAI Whisper (tiny, 39M params) to transcribe audio files to Hindi text.
Model is loaded once and cached for the session lifetime.
"""

import whisper
import numpy as np
import sys
import os

# Add project root to path so `config` can be imported from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

_model = None  # Module-level cache — load once, reuse many times


def _get_model():
    """Lazy-load Whisper model (downloads ~150 MB on first run, cached after)."""
    global _model
    if _model is None:
        print(f"[transcribe] Loading Whisper-{config.WHISPER_MODEL_SIZE} model …", flush=True)
        _model = whisper.load_model(config.WHISPER_MODEL_SIZE)
        print("[transcribe] Model ready.", flush=True)
    return _model


def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to Hindi text.

    Args:
        audio_path: Absolute or relative path to a WAV/MP3/M4A file.

    Returns:
        Transcribed Hindi text string (may be empty if audio is silent/unintelligible).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    # Force language=Hindi and suppress other-language fallback
    result = model.transcribe(
        audio_path,
        language=config.WHISPER_LANGUAGE,
        task="transcribe",
        fp16=False,          # CPU-safe
        verbose=False,
    )

    transcript = result["text"].strip()
    return transcript


def transcribe_array(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe a raw numpy audio array (float32, mono) to Hindi text.
    Used by the Streamlit app which records directly to a numpy array.

    Args:
        audio_array: 1-D float32 numpy array at 16 kHz.
        sample_rate: Must be 16000 for Whisper.

    Returns:
        Transcribed Hindi text string.
    """
    import tempfile, soundfile as sf

    # Whisper works best from a file; save to a temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio_array, sample_rate)
    try:
        result = transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result


if __name__ == "__main__":
    # Quick smoke test: python src/transcribe.py <audio_file>
    if len(sys.argv) < 2:
        print("Usage: python src/transcribe.py <audio_file>")
        sys.exit(1)
    text = transcribe(sys.argv[1])
    print(f"Transcript: {text}")
