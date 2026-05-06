import os
import sys
import json
import shutil
import argparse
from pathlib import Path

# Add the project root to sys.path so we can import config and src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.matcher import match_command

def mine_local_fleurs():
    data_dir = Path(__file__).parent
    train_dir = data_dir / "train"
    train_tsv = data_dir / "train.tsv"
    
    if not train_dir.exists() or not train_tsv.exists():
        print(f"ERROR: Could not find 'train' directory or 'train.tsv' in {data_dir}")
        sys.exit(1)

    mined_dir = data_dir / "mined_local_fleurs"
    
    # Create output directories for each command
    for cmd in config.COMMANDS:
        (mined_dir / cmd).mkdir(parents=True, exist_ok=True)

    print("Mining local FLEURS dataset...")
    
    match_counts = {cmd: 0 for cmd in config.COMMANDS}
    mined_manifest = []
    scanned = 0
    total_matches_found = 0

    with open(train_tsv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            # FLEURS train.tsv format typically:
            # id \t filename \t raw_transcript \t transcript \t ...
            if len(parts) < 4:
                continue
                
            sample_id = parts[0]
            filename = parts[1]
            transcript = parts[3]  # The normalized transcript without punctuation
            
            scanned += 1
            
            # Test against our keywords
            result = match_command(transcript)
            
            if result["command"] != "unknown" and result["method"] == "keyword":
                cmd = result["command"]
                
                source_audio = train_dir / filename
                if not source_audio.exists():
                    # Fallback just in case filename doesn't match perfectly
                    continue
                    
                target_audio = mined_dir / cmd / filename
                
                # Copy the audio file
                shutil.copy2(source_audio, target_audio)
                
                mined_manifest.append({
                    "original_id": sample_id,
                    "file": str(Path(cmd) / filename),
                    "transcript": transcript,
                    "command": cmd
                })
                
                match_counts[cmd] += 1
                total_matches_found += 1
                print(f"[{cmd}] Found match #{total_matches_found}: '{transcript}'")
                
            if scanned % 1000 == 0:
                print(f"  ... scanned {scanned} samples so far (Found {total_matches_found} matches) ...")

    # Save the mined manifest
    if mined_manifest:
        with open(mined_dir / "mined_manifest.json", "w", encoding="utf-8") as f:
            json.dump(mined_manifest, f, ensure_ascii=False, indent=2)

    print("\n--- Mining Complete ---")
    print(f"Scanned {scanned} total samples.")
    print(f"Saved {total_matches_found} matching clips to {mined_dir}.")
    for cmd, count in match_counts.items():
        if count > 0:
            print(f"  {cmd}: {count} clips")

if __name__ == "__main__":
    mine_local_fleurs()
