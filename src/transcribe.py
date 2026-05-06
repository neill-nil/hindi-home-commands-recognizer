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

from pathlib import Path

_model = None  # Module-level cache — load once, reuse many times
_is_hf_model = False

def _get_model():
    """Lazy-load Whisper model. Prefers local HF finetuned model if present."""
    global _model, _is_hf_model
    if _model is None:
        finetuned_path = Path(config.BASE_DIR) / "models" / "whisper_finetuned"
        if finetuned_path.exists():
            print(f"[transcribe] Loading FINE-TUNED HuggingFace model from {finetuned_path}...", flush=True)
            from transformers import pipeline
            # Using CPU/MPS automatically mapped by HF pipeline
            _model = pipeline("automatic-speech-recognition", model=str(finetuned_path))
            _is_hf_model = True
        else:
            print(f"[transcribe] Loading base openai/Whisper-{config.WHISPER_MODEL_SIZE} model...", flush=True)
            _model = whisper.load_model(config.WHISPER_MODEL_SIZE)
            _is_hf_model = False
        print("[transcribe] Model ready.", flush=True)
    return _model

def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file to Hindi text.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    if _is_hf_model:
        # Generate transcription using HF pipeline
        result = model(audio_path, generate_kwargs={"language": "hi", "task": "transcribe"})
        transcript = result["text"].strip()
    else:
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
