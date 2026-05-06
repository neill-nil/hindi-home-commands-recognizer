"""
src/train_classifier.py — Train audio classifier with richer features.

Reads data/self_recorded/<command>/*.wav, extracts MFCC+delta+chroma features,
trains a Gradient Boosting + SVM ensemble, evaluates with cross-validation.

Usage:
    python3 src/train_classifier.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from pathlib import Path
from collections import Counter

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

import config
from src.features import extract_features_from_file, CLIP_DURATION

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}
MODEL_DIR  = Path(config.BASE_DIR) / "models"


def load_dataset():
    """Load all self-recorded clips and extract features."""
    data_dir = Path(config.SELF_RECORDED_DIR)
    X, y = [], []

    print(f"\nLoading clips from: {data_dir}")
    print(f"  Clip duration (pad/trim): {CLIP_DURATION}s\n")

    per_class = Counter()
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in config.COMMANDS:
            continue
        for f in sorted(label_dir.glob("*")):
            if f.suffix.lower() not in AUDIO_EXTS:
                continue
            feats = extract_features_from_file(str(f))
            if feats is not None:
                X.append(feats)
                y.append(label_dir.name)
                per_class[label_dir.name] += 1

    for cmd in config.COMMANDS:
        n = per_class.get(cmd, 0)
        status = "✓" if n > 0 else "✗ MISSING"
        print(f"  {status}  {cmd:<15}  {n} clips")

    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("   Audio Classifier Training — Rich Feature Edition")
    print("=" * 60)

    X, y = load_dataset()
    if len(X) == 0:
        print("\nNo data found! Record clips first with data/record_audio.py")
        sys.exit(1)

    print(f"\nTotal: {len(X)} clips  ·  Feature dim: {X.shape[1]}\n")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Cross-validation to check model quality ──────────────────────────────
    print("Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # SVM with Platt scaling for well-calibrated probabilities
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=10, gamma="scale"), cv=3
    )
    cv_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"  SVM (calibrated) CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                      max_depth=4, random_state=42)
    cv_scores_gbc = cross_val_score(gbc, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"  Gradient Boosting     CV accuracy: {cv_scores_gbc.mean()*100:.1f}% ± {cv_scores_gbc.std()*100:.1f}%")

    # ── Train final ensemble on ALL data ─────────────────────────────────────
    print("\nTraining final ensemble on all data …")
    final_svm = CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), cv=3)
    final_gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                           max_depth=4, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("svm", final_svm), ("gbc", final_gbc)],
        voting="soft",
        weights=[1, 1],
    )
    ensemble.fit(X_scaled, y)

    # Quick train-set sanity check
    y_pred = ensemble.predict(X_scaled)
    print(f"  Train accuracy (sanity): {accuracy_score(y, y_pred)*100:.1f}%")

    print("\nPer-class report (on training data — use CV score for real estimate):")
    print(classification_report(y, y_pred, target_names=sorted(set(y))))

    # ── Save ─────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(ensemble, MODEL_DIR / "audio_svm.pkl")
    joblib.dump(scaler,   MODEL_DIR / "audio_scaler.pkl")

    print("=" * 60)
    print(f"Saved to {MODEL_DIR}/")
    print(f"Restart the Streamlit app to use the new model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
