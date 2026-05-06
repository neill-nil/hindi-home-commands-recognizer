"""
src/train_whisper.py — Fine-tune Whisper encoder for Hindi command classification

Pipeline:
  1. Load self-recorded WAV clips
  2. Extract mel-spectrograms using WhisperFeatureExtractor (same as Whisper uses internally)
  3. Pass through frozen Whisper encoder → global average pool → MLP head
  4. Train head on 240 clips, 80/20 split
  5. Optionally unfreeze last encoder layers for deeper fine-tuning
  6. Save model weights to models/

Usage:
    python3 src/train_whisper.py
    python3 src/train_whisper.py --epochs 30 --unfreeze 2
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import WhisperFeatureExtractor
from pathlib import Path
from collections import Counter
import librosa
import joblib

import config
from src.whisper_classifier import WhisperCommandClassifier, HF_MODEL_ID, SAMPLE_RATE, CLIP_DURATION

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}
MODEL_DIR  = Path(config.BASE_DIR) / "models"


# ── Dataset ────────────────────────────────────────────────────────────────────
class CommandDataset(Dataset):
    def __init__(self, files, labels, label2idx, feature_extractor):
        self.files   = files
        self.labels  = labels
        self.l2i     = label2idx
        self.fx      = feature_extractor
        self.target_len = int(SAMPLE_RATE * CLIP_DURATION)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.labels[idx]

        # Load and normalize
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        peak  = np.abs(y).max()
        if peak > 1e-6:
            y = y / peak

        # Pad / trim to fixed length
        if len(y) >= self.target_len:
            y = y[:self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))

        # WhisperFeatureExtractor → 80-bin log-mel spectrogram
        feats = self.fx(y.astype(np.float32), sampling_rate=SAMPLE_RATE,
                        return_tensors="pt").input_features[0]  # (80, T)

        return feats, self.l2i[label]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_files():
    data_dir = Path(config.SELF_RECORDED_DIR)
    files, labels = [], []
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in config.COMMANDS:
            continue
        for f in sorted(label_dir.glob("*")):
            if f.suffix.lower() in AUDIO_EXTS:
                files.append(str(f))
                labels.append(label_dir.name)
    return files, labels


# ── Training ───────────────────────────────────────────────────────────────────
def train(epochs=25, unfreeze_last_n=0, batch_size=8, lr=3e-4):
    device = torch.device("cpu")   # CPU-capable
    print(f"\n{'='*60}")
    print(f"  Whisper-tiny Fine-Tuning — {epochs} epochs, unfreeze={unfreeze_last_n}")
    print(f"{'='*60}\n")

    # Load data
    files, labels = load_files()
    if not files:
        print("ERROR: No audio files found. Record clips first!")
        sys.exit(1)

    all_commands = sorted(set(labels))
    label2idx    = {c: i for i, c in enumerate(all_commands)}
    idx2label    = {i: c for c, i in label2idx.items()}
    num_classes  = len(all_commands)

    print(f"Commands ({num_classes}): {all_commands}")
    print(f"Total clips: {len(files)}\n")

    # 80 / 20 stratified split
    from sklearn.model_selection import train_test_split
    tr_files, va_files, tr_labels, va_labels = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(tr_files)}  |  Val: {len(va_files)}\n")

    # Feature extractor (Whisper's mel pipeline)
    fx = WhisperFeatureExtractor.from_pretrained(HF_MODEL_ID)

    tr_ds = CommandDataset(tr_files, tr_labels, label2idx, fx)
    va_ds = CommandDataset(va_files, va_labels, label2idx, fx)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = WhisperCommandClassifier(num_classes=num_classes, unfreeze_last_n=unfreeze_last_n)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({trainable/total*100:.1f}%)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state   = None

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>8}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for feats, targets in tr_dl:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss   += loss.item() * len(targets)
            tr_correct += (logits.argmax(1) == targets).sum().item()
            tr_total   += len(targets)
        scheduler.step()

        # ── Validate ──
        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for feats, targets in va_dl:
                feats, targets = feats.to(device), targets.to(device)
                logits = model(feats)
                va_correct += (logits.argmax(1) == targets).sum().item()
                va_total   += len(targets)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        avg_loss = tr_loss / tr_total

        marker = " ◀ best" if va_acc > best_val_acc else ""
        print(f"{epoch:>6} | {avg_loss:>10.4f} | {tr_acc:>8.1%} | {va_acc:>7.1%}{marker}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save best model ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    model.load_state_dict(best_state)

    torch.save({
        "state_dict":  model.state_dict(),
        "label2idx":   label2idx,
        "idx2label":   idx2label,
        "num_classes": num_classes,
        "unfreeze_last_n": unfreeze_last_n,
    }, MODEL_DIR / "whisper_classifier.pt")

    print(f"\n✓  Best val accuracy: {best_val_acc:.1%}")
    print(f"✓  Saved to {MODEL_DIR}/whisper_classifier.pt")
    print(f"\nRestart the Streamlit app — it will auto-detect and use the Whisper classifier.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=25, help="Training epochs")
    parser.add_argument("--unfreeze", type=int, default=0,
                        help="Unfreeze last N encoder blocks (0=head only, 2=recommended)")
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--lr",       type=float, default=3e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, unfreeze_last_n=args.unfreeze,
          batch_size=args.batch, lr=args.lr)
