"""
data/record_audio.py — CLI helper to record self-recorded command clips

Usage:
    python data/record_audio.py --student YOUR_NAME

For each of the 8 commands, press Enter to start 3-second recording,
again to stop. Repeats for 10 takes per command.
Saves to:  data/self_recorded/<command>/<student>_<take_num>.wav
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path

import config

SAMPLE_RATE   = 16000   # Hz  — Whisper expects 16kHz
DURATION_SECS = 3       # Max seconds per recording
N_TAKES       = 10      # Takes per command per student


def record_clip(duration: float = DURATION_SECS, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record a single audio clip of `duration` seconds."""
    print(f"  🎙  Recording for {duration}s … (speak now)", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser(description="Record Hindi command clips")
    parser.add_argument("--student", required=True,
                        help="Your name or ID (used in filename, e.g. 'neil')")
    parser.add_argument("--takes",   type=int, default=N_TAKES,
                        help=f"Number of takes per command (default {N_TAKES})")
    parser.add_argument("--duration", type=float, default=DURATION_SECS,
                        help=f"Recording duration in seconds (default {DURATION_SECS})")
    args = parser.parse_args()

    student = args.student.strip().replace(" ", "_")

    print(f"\n{'='*55}")
    print(f" Hindi Command Recorder — Student: {student}")
    print(f" {args.takes} takes × {len(config.COMMANDS)} commands = "
          f"{args.takes * len(config.COMMANDS)} total clips")
    print(f"{'='*55}\n")

    for command in config.COMMANDS:
        canonical = config.CANONICAL_PHRASES[command]
        out_dir   = Path(config.SELF_RECORDED_DIR) / command
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── Command: {command}  ({canonical}) ──")
        print(f"   Say the phrase each time you're prompted.\n")

        for take in range(1, args.takes + 1):
            out_file = out_dir / f"{student}_{take:02d}.wav"

            if out_file.exists():
                print(f"  [skip] Take {take:02d} already exists: {out_file.name}")
                continue

            input(f"  [ Take {take:02d}/{args.takes} ]  Press Enter to record … ")
            audio = record_clip(args.duration)

            # Save
            sf.write(str(out_file), audio, SAMPLE_RATE)
            print(f"  ✓  Saved: {out_file}")
            time.sleep(0.5)

    print(f"\n🎉  Done! All clips saved under data/self_recorded/")


if __name__ == "__main__":
    main()
