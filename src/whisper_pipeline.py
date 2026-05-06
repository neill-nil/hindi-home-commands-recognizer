"""
src/whisper_pipeline.py — Inference using fine-tuned Whisper classifier

Loads models/whisper_classifier.pt (saved by train_whisper.py) and
predicts command labels directly from audio without any text transcription.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor
from pathlib import Path
import librosa

import config
from src.whisper_classifier import WhisperCommandClassifier, HF_MODEL_ID, SAMPLE_RATE, CLIP_DURATION

MODEL_PATH = Path(config.BASE_DIR) / "models" / "whisper_classifier.pt"
CONFIDENCE_THRESHOLD = 0.35

_model    = None
_fx       = None
_idx2label = None


def load_whisper_model() -> bool:
    global _model, _fx, _idx2label
    if _model is not None:
        return True
    if not MODEL_PATH.exists():
        return False
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _idx2label = ckpt["idx2label"]
        _model = WhisperCommandClassifier(
            num_classes=ckpt["num_classes"],
            unfreeze_last_n=ckpt.get("unfreeze_last_n", 0),
        )
        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()
        _fx = WhisperFeatureExtractor.from_pretrained(HF_MODEL_ID)
        print("[whisper_pipeline] Loaded whisper_classifier.pt")
        return True
    except Exception as e:
        print(f"[whisper_pipeline] Failed to load model: {e}")
        return False


def _preprocess(y: np.ndarray, sr: int) -> torch.Tensor:
    """Convert raw audio to Whisper mel-spectrogram tensor."""
    # Resample if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    # Normalize
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak
    # Pad / trim
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(y) >= target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target - len(y)))
    feats = _fx(y.astype(np.float32), sampling_rate=SAMPLE_RATE,
                return_tensors="pt").input_features   # (1, 80, T)
    return feats


def predict_whisper(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Predict command from a raw float32 numpy audio array."""
    if not load_whisper_model():
        return {"command": "unknown", "confidence": 0.0,
                "method": "whisper_missing", "transcript": "[Run train_whisper.py first]"}
    try:
        feats = _preprocess(audio_array, sample_rate)
        with torch.no_grad():
            logits = _model(feats)
            probs  = F.softmax(logits, dim=-1)[0]
        top_idx = probs.argmax().item()
        conf    = probs[top_idx].item()
        command = _idx2label[top_idx]

        if conf < CONFIDENCE_THRESHOLD:
            return {"command": "unknown", "confidence": conf,
                    "method": f"whisper_low_conf ({command} @ {conf:.2f})",
                    "transcript": "[Below confidence threshold]"}

        return {"command": command, "confidence": conf,
                "method": "whisper_finetuned", "transcript": f"[Whisper encoder → {command}]"}
    except Exception as e:
        print(f"[whisper_pipeline] predict error: {e}")
        return {"command": "unknown", "confidence": 0.0, "method": "error", "transcript": ""}


def predict_whisper_from_file(path: str) -> dict:
    """Predict command from a WAV/MP3 file."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    return predict_whisper(y, sr)
