"""
audio_pipeline.py — Pure Audio ML Pipeline (No Whisper)

Extracts MFCCs from the live audio buffer and predicts directly using the trained SVM.
"""

import os
import sys
import numpy as np
import librosa
import joblib
from pathlib import Path
import tempfile
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

MODEL_DIR = Path(config.BASE_DIR) / "models"
SVM_PATH = MODEL_DIR / "audio_svm.pkl"
SCALER_PATH = MODEL_DIR / "audio_scaler.pkl"

# Load models globally so it's snappy
_clf = None
_scaler = None

def load_models():
    global _clf, _scaler
    if _clf is None and SVM_PATH.exists() and SCALER_PATH.exists():
        _clf = joblib.load(SVM_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _clf is not None

def predict_audio_direct(audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Given a raw numpy audio array, extract MFCCs and predict directly using the local SVM model.
    """
    if not load_models():
        return {"command": "unknown", "confidence": 0.0, "method": "ml_missing", "transcript": "[Model not trained]"}
    
    try:
        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Scale and predict
        features_scaled = _scaler.transform([mfccs_mean])
        
        # Get top class and probability
        probs = _clf.predict_proba(features_scaled)[0]
        top_idx = np.argmax(probs)
        command = _clf.classes_[top_idx]
        conf = probs[top_idx]
        
        # If low confidence, return unknown
        if conf < 0.4:
            return {
                "command": "unknown", 
                "confidence": conf, 
                "method": "ml_audio_low_conf", 
                "transcript": "[Audio unrecognized]"
            }
            
        return {
            "command": command,
            "confidence": conf,
            "method": "ml_audio",
            "transcript": "[Classified directly from Audio Profile]"
        }
    except Exception as e:
        print(f"Error in ML pipeline predicting: {e}")
        return {"command": "unknown", "confidence": 0.0, "method": "error", "transcript": ""}

def predict_from_file_direct(audio_path: str) -> dict:
    """
    Predict command directly from wav file.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    return predict_audio_direct(y, sr)
