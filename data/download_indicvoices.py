"""
data/download_indicvoices.py — Download Hindi subset from ai4bharat/IndicVoices

This script:
  1. Loads the Hindi split of IndicVoices from HuggingFace (gated, needs token)
  2. Saves audio files + transcripts to data/indicvoices_hindi/

Requirements:
  - Run `huggingface-cli login` first OR set env var HF_TOKEN=<your_token>

Usage:
    python data/download_indicvoices.py
    python data/download_indicvoices.py --max_samples 500
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import config

OUTPUT_DIR = Path(config.INDICVOICES_DIR)


def download(max_samples: int = None):
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("ERROR: Run `pip install datasets soundfile` first.")
        sys.exit(1)

    print("Loading ai4bharat/IndicVoices (Hindi) from HuggingFace …")
    print("NOTE: This is a GATED dataset. You must have accepted the terms first.")
    print("      If you get an auth error, run:  huggingface-cli login\n")

    try:
        # IndicVoices uses language-specific configs; Hindi config is "hi"
        dataset = load_dataset(
            "ai4bharat/IndicVoices",
            "hi",                  # Hindi config
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\nFailed to load dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Accept dataset terms at https://huggingface.co/datasets/ai4bharat/IndicVoices")
        print("  2. Run: huggingface-cli login")
        print("  3. Make sure you have `datasets` installed: pip install datasets")
        sys.exit(1)

    total = len(dataset)
    if max_samples:
        total = min(total, max_samples)
    print(f"Dataset loaded. Processing {total} samples …\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        sample_rate = sample["audio"]["sampling_rate"]
        transcript  = sample.get("transcript", sample.get("text", ""))

        # Save audio
        out_wav = OUTPUT_DIR / f"sample_{i:05d}.wav"
        sf.write(str(out_wav), audio_array, sample_rate)

        manifest.append({
            "id": i,
            "file": str(out_wav.name),
            "transcript": transcript,
            "sampling_rate": sample_rate,
        })

        if (i + 1) % 100 == 0:
            print(f"  Saved {i+1}/{total} …")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✓  Done. {len(manifest)} files saved to {OUTPUT_DIR}")
    print(f"   Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IndicVoices Hindi subset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to download (default: all)")
    args = parser.parse_args()
    download(args.max_samples)
