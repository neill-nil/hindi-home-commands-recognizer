"""
src/evaluate.py — Accuracy evaluation on self-recorded audio clips

Folder structure expected:
    data/self_recorded/
        light_on/   student1_01.wav  student1_02.wav …
        light_off/  …
        …

Usage:
    python src/evaluate.py
    python src/evaluate.py --data_dir data/self_recorded
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path
from collections import defaultdict

import config
from src.pipeline import predict


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def evaluate(data_dir: str) -> dict:
    """
    Run the pipeline on all audio files in data_dir and compute metrics.

    Args:
        data_dir: Root directory with one sub-folder per command label.

    Returns:
        results dict with per-class accuracy and overall accuracy.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Collect all audio files and their ground-truth labels from folder names
    samples = []
    for label_dir in sorted(data_path.iterdir()):
        if not label_dir.is_dir():
            continue
        gt_label = label_dir.name
        if gt_label not in config.COMMANDS:
            print(f"  [warn] Skipping unknown folder: {gt_label}")
            continue
        for audio_file in sorted(label_dir.glob("*")):
            if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((str(audio_file), gt_label))

    if not samples:
        print(f"No audio files found in {data_dir}")
        return {}

    print(f"\nEvaluating {len(samples)} clips …\n")

    # Per-class counters
    correct_per_class   = defaultdict(int)
    total_per_class     = defaultdict(int)
    confusion           = defaultdict(lambda: defaultdict(int))
    all_results         = []

    for audio_path, gt_label in samples:
        try:
            pred = predict(audio_path)
        except Exception as e:
            print(f"  [error] {audio_path}: {e}")
            pred = {"command": "error", "confidence": 0.0, "method": "error", "transcript": ""}

        pred_label = pred["command"]
        is_correct = (pred_label == gt_label)

        total_per_class[gt_label]  += 1
        correct_per_class[gt_label] += int(is_correct)
        confusion[gt_label][pred_label] += 1

        status = "✓" if is_correct else "✗"
        print(f"  {status}  [{gt_label}]  →  {pred_label}  |  \"{pred['transcript'][:50]}\"")

        all_results.append({
            "file": audio_path,
            "gt": gt_label,
            "pred": pred_label,
            "correct": is_correct,
            "transcript": pred["transcript"],
            "confidence": pred["confidence"],
        })

    # Print summary
    print("\n" + "="*60)
    print("Per-class Accuracy")
    print("="*60)
    total_correct = 0
    total_count   = 0
    for cmd in config.COMMANDS:
        n     = total_per_class[cmd]
        c     = correct_per_class[cmd]
        acc   = (c / n * 100) if n > 0 else 0.0
        total_correct += c
        total_count   += n
        print(f"  {cmd:<15}  {c:>3}/{n:<3}  ({acc:5.1f}%)")

    overall = (total_correct / total_count * 100) if total_count > 0 else 0.0
    print("-"*60)
    print(f"  {'OVERALL':<15}  {total_correct:>3}/{total_count:<3}  ({overall:5.1f}%)")

    # Print confusion matrix
    print("\nConfusion Matrix (rows=GT, cols=Pred)")
    header = [""] + config.COMMANDS + ["unknown"]
    print("  " + "  ".join(f"{h[:8]:>8}" for h in header))
    for gt in config.COMMANDS:
        row = [gt[:8]]
        for pred_lbl in config.COMMANDS + ["unknown"]:
            row.append(str(confusion[gt][pred_lbl]))
        print("  " + "  ".join(f"{v:>8}" for v in row))

    results = {
        "overall_accuracy": overall,
        "per_class_accuracy": {
            cmd: (correct_per_class[cmd] / total_per_class[cmd] * 100)
                 if total_per_class[cmd] > 0 else 0.0
            for cmd in config.COMMANDS
        },
        "total_samples": total_count,
        "details": all_results,
    }

    # Save results JSON
    out_path = "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hindi command recognizer")
    parser.add_argument("--data_dir", default=config.SELF_RECORDED_DIR,
                        help="Root dir with command sub-folders")
    args = parser.parse_args()
    evaluate(args.data_dir)
