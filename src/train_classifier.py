"""
train_classifier.py — Train a traditional ML classifier on MFCC audio features.

This completely bypasses Whisper and trains specifically on your team's voices.
It extracts MFCCs from `data/self_recorded` and trains a Random Forest classifier.

Usage:
    python src/train_classifier.py
"""

import os
import sys
import numpy as np
import librosa
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def extract_features(audio_path, sr=16000):
    """
    Load an audio file and extract its mean MFCC features.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        
        # If audio is empty or too short
        if len(y) == 0:
            return None
            
        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Average across the time axis to get a 1D feature vector for the whole clip
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    print("="*60)
    print("       Training Direct Audio Classifier (MFCCs)")
    print("="*60)
    
    data_dir = Path(config.SELF_RECORDED_DIR)
    if not data_dir.exists():
        print(f"Error: Directory not found - {data_dir}")
        print("Please record your clips first!")
        sys.exit(1)

    X = []
    y = []
    
    print("Extracting features from self_recorded folder...")
    
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
            
        label = label_dir.name
        if label not in config.COMMANDS:
            continue
            
        audio_files = list(label_dir.glob("*.wav"))
        for file in audio_files:
            features = extract_features(str(file))
            if features is not None:
                X.append(features)
                y.append(label)
                
    if not X:
        print("No audio data found! Record clips first.")
        sys.exit(1)
        
    X = np.array(X)
    y = np.array(y)
    print(f"\nExtracted features from {len(X)} clips across {len(set(y))} classes.")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features (very important for SVM / ML models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a fast SVM or Random Forest
    print("\nTraining Classifier (SVM)...")
    clf = SVC(kernel='rbf', probability=True, C=10) # SVM usually does great on MFCCs
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    print("\nEvaluating on 20% hold-out test set:")
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Hold-out Accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_test, y_pred))

    # Save models
    model_dir = Path(config.BASE_DIR) / "models"
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(clf, model_dir / "audio_svm.pkl")
    joblib.dump(scaler, model_dir / "audio_scaler.pkl")
    
    print("="*60)
    print(f"Models saved to {model_dir}/")
    print("They will now be automatically used by the Streamlit App!")
    print("="*60)

if __name__ == "__main__":
    main()
