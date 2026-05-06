"""
src/features.py — Shared audio feature extraction used by BOTH training and inference.

Having one shared module guarantees training and inference use EXACTLY the same features.
"""

import numpy as np
import librosa

# ── Must match the MAX_DURATION in app.py ─────────────────────────────────────
CLIP_DURATION = 4       # seconds — pad/trim everything to this
SAMPLE_RATE   = 16000
N_MFCC        = 40


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1] so volume differences don't matter."""
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak
    return y


def pad_or_trim(y: np.ndarray, sr: int = SAMPLE_RATE, duration: float = CLIP_DURATION) -> np.ndarray:
    """Ensure every clip is exactly `duration` seconds long."""
    target_len = int(sr * duration)
    if len(y) >= target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a rich, fixed-size feature vector from a pre-loaded audio array.

    Features (in order):
      - 40 MFCC means
      - 40 MFCC stds         ← captures dynamics, not just average timbre
      - 40 delta-MFCC means  ← 1st-order temporal differences
      - 40 delta²-MFCC means ← 2nd-order temporal differences
      - 12 chroma means      ← pitch class profile (helps on/off discrimination)
      - 1  RMS energy mean   ← overall loudness
      Total: 173-dim vector
    """
    y = normalize_audio(y)
    y = pad_or_trim(y, sr)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    # Delta MFCCs (temporal derivatives)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    d1_mean = np.mean(delta1, axis=1)
    d2_mean = np.mean(delta2, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    return np.concatenate([mfcc_mean, mfcc_std, d1_mean, d2_mean, chroma_mean, [rms_mean]])


def extract_features_from_file(path: str) -> np.ndarray | None:
    """Load a file and extract features. Returns None on error."""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        return extract_features(y, sr)
    except Exception as e:
        print(f"  [warn] Could not process {path}: {e}")
        return None
