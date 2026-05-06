"""
src/audio_pipeline.py — Pure Audio ML Pipeline (No Whisper)

Predicts directly using the trained SVM/ensemble via the shared features.py extractor.
Training and inference now use EXACTLY the same feature extraction logic.
"""

import os, sys
import numpy as np
import librosa
import joblib
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.features import extract_features, extract_features_from_file

MODEL_DIR   = Path(config.BASE_DIR) / "models"
SVM_PATH    = MODEL_DIR / "audio_svm.pkl"
SCALER_PATH = MODEL_DIR / "audio_scaler.pkl"

# Confidence threshold — below this → "unknown"
# Lowered to 0.25 because calibrated probabilities are better distributed
CONFIDENCE_THRESHOLD = 0.25

_clf    = None
_scaler = None


def load_models() -> bool:
    global _clf, _scaler
    if _clf is None and SVM_PATH.exists() and SCALER_PATH.exists():
        _clf    = joblib.load(SVM_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _clf is not None


def predict_audio_direct(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Predict command from a raw float32 numpy audio array.
    Uses the shared features.py extractor — same as training time.
    """
    if not load_models():
        return {"command": "unknown", "confidence": 0.0,
                "method": "ml_missing", "transcript": "[Model not trained — run train_classifier.py]"}

    try:
        feats = extract_features(audio_array, sample_rate)   # 173-dim vector
        X = _scaler.transform([feats])

        probs   = _clf.predict_proba(X)[0]
        top_idx = np.argmax(probs)
        command = _clf.classes_[top_idx]
        conf    = float(probs[top_idx])

        if conf < CONFIDENCE_THRESHOLD:
            return {
                "command":    "unknown",
                "confidence": conf,
                "method":     f"ml_low_conf ({command} @ {conf:.2f})",
                "transcript": "[Below confidence threshold]",
            }

        return {
            "command":    command,
            "confidence": conf,
            "method":     "ml_audio",
            "transcript": "[Direct audio classification]",
        }
    except Exception as e:
        print(f"[audio_pipeline] Error: {e}")
        return {"command": "unknown", "confidence": 0.0, "method": "error", "transcript": ""}


def predict_from_file_direct(audio_path: str) -> dict:
    """Predict command from a WAV/MP3 file path."""
    if not load_models():
        return {"command": "unknown", "confidence": 0.0,
                "method": "ml_missing", "transcript": "[Model not trained]"}

    try:
        feats = extract_features_from_file(audio_path)
        if feats is None:
            return {"command": "unknown", "confidence": 0.0,
                    "method": "error", "transcript": ""}
        X = _scaler.transform([feats])

        probs   = _clf.predict_proba(X)[0]
        top_idx = np.argmax(probs)
        command = _clf.classes_[top_idx]
        conf    = float(probs[top_idx])

        if conf < CONFIDENCE_THRESHOLD:
            return {"command": "unknown", "confidence": conf,
                    "method": f"ml_low_conf ({command})", "transcript": ""}

        return {"command": command, "confidence": conf,
                "method": "ml_audio", "transcript": "[Direct audio classification]"}
    except Exception as e:
        print(f"[audio_pipeline] File prediction error: {e}")
        return {"command": "unknown", "confidence": 0.0, "method": "error", "transcript": ""}
