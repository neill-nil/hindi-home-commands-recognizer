"""
src/pipeline.py — End-to-end audio → command prediction pipeline

Chains transcription and matching into a single predict() call.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.transcribe import transcribe, transcribe_array
from src.matcher import match_command


def predict(audio_path: str) -> dict:
    """
    Predict smart-home command from an audio file.

    Args:
        audio_path: Path to WAV/MP3/M4A audio file.

    Returns:
        dict with keys: command, confidence, method, transcript
    """
    transcript = transcribe(audio_path)
    result = match_command(transcript)
    return result


def predict_array(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Predict smart-home command from a raw audio numpy array.
    Used by the Streamlit real-time recording flow.

    Args:
        audio_array: 1-D float32 numpy array at 16 kHz.
        sample_rate: Must be 16000.

    Returns:
        dict with keys: command, confidence, method, transcript
    """
    transcript = transcribe_array(audio_array, sample_rate)
    result = match_command(transcript)
    return result


if __name__ == "__main__":
    # Quick smoke test: python src/pipeline.py <audio_file>
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <audio_file>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"Transcript : {result['transcript']}")
    print(f"Command    : {result['command']}")
    print(f"Confidence : {result['confidence']:.2f}")
    print(f"Method     : {result['method']}")
