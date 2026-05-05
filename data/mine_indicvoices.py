"""
data/mine_indicvoices.py — Mine IndicVoices for smart-home commands

This script searches the IndicVoices dataset (downloaded via download_indicvoices.py)
for transcripts that match our smart-home keywords. If an exact keyword match is found, 
it copies the audio clip to a new categorized dataset folder to be used for fine-tuning.

Usage:
    python data/mine_indicvoices.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import shutil
from pathlib import Path

import config
from src.matcher import match_command

def mine_dataset():
    manifest_path = Path(config.INDICVOICES_DIR) / "manifest.json"
    mined_dir = Path(config.DATA_DIR) / "mined_indicvoices"
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please run 'python data/download_indicvoices.py' first to get the dataset.")
        sys.exit(1)
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print(f"Loaded {len(manifest)} samples from IndicVoices manifest.")
    print(f"Mining for {len(config.COMMANDS)} smart-home commands...\n")
    
    # Create output directories for each command
    for cmd in config.COMMANDS:
        (mined_dir / cmd).mkdir(parents=True, exist_ok=True)
        
    match_counts = {cmd: 0 for cmd in config.COMMANDS}
    mined_manifest = []
    
    for entry in manifest:
        transcript = entry.get("transcript", "")
        audio_filename = entry.get("file")
        source_audio = Path(config.INDICVOICES_DIR) / audio_filename
        
        # We reuse your existing string matcher!
        # We ONLY want exact keyword matches ("method" == "keyword"). 
        # We ignore fuzzy matches to ensure our training data is 100% accurate ground-truth.
        result = match_command(transcript)
        
        if result["command"] != "unknown" and result["method"] == "keyword":
            cmd = result["command"]
            
            # Verify source audio actually exists
            if not source_audio.exists():
                continue
                
            # Copy to our categorized mined directory
            dest_audio = mined_dir / cmd / f"indicvoices_{audio_filename}"
            shutil.copy2(source_audio, dest_audio)
            
            # Record in our new specialized manifest
            mined_manifest.append({
                "original_id": entry.get("id"),
                "file": str(Path(cmd) / f"indicvoices_{audio_filename}"),
                "transcript": transcript,
                "command": cmd
            })
            
            match_counts[cmd] += 1
            print(f"[{cmd}] Found match: '{transcript}'")

    # Save the mined manifest
    if mined_manifest:
        with open(mined_dir / "mined_manifest.json", "w", encoding="utf-8") as f:
            json.dump(mined_manifest, f, ensure_ascii=False, indent=2)
            
    print("\n--- Mining Complete ---")
    total_matches = sum(match_counts.values())
    print(f"Found {total_matches} total matching clips.")
    for cmd, count in match_counts.items():
        print(f"  {cmd}: {count} clips")
    print(f"\nMined dataset saved to: {mined_dir}")

if __name__ == "__main__":
    mine_dataset()
