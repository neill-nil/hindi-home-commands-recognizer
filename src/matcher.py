"""
src/matcher.py — Rule-based + fuzzy string matcher

Maps a Hindi transcript to one of 8 command labels using:
  1. Keyword scoring: check for device word + state word in transcript
  2. Fuzzy match fallback: rapidfuzz similarity vs canonical phrases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from rapidfuzz import fuzz, process


def _keyword_match(transcript: str) -> tuple[str, float]:
    """
    Score each command by counting keyword hits in the transcript.

    Returns:
        (best_command, confidence_score) where score = device_hit * state_hit.
        Returns ("unknown", 0.0) if no command scores positively.
    """
    t_lower = transcript.lower()
    best_cmd = "unknown"
    best_score = 0.0

    for command, (device_words, state_words) in config.KEYWORD_MAP.items():
        device_hit = any(dw.lower() in t_lower for dw in device_words)
        state_hit  = any(sw.lower() in t_lower for sw in state_words)

        if device_hit and state_hit:
            # Both matched — strong signal
            score = 1.0
        elif device_hit or state_hit:
            # Partial match — weak signal
            score = 0.5
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_cmd = command

    return best_cmd, best_score


def _fuzzy_match(transcript: str) -> tuple[str, float]:
    """
    Find best command via rapidfuzz token-set ratio against canonical phrases.

    Returns:
        (best_command, normalized_confidence 0.0–1.0)
    """
    choices = {cmd: phrase for cmd, phrase in config.CANONICAL_PHRASES.items()}

    best_cmd = "unknown"
    best_score = 0.0

    for cmd, phrase in choices.items():
        score = fuzz.token_set_ratio(transcript, phrase)   # 0–100
        if score > best_score:
            best_score = score
            best_cmd = cmd

    # Normalize to 0.0–1.0 and apply threshold
    norm_score = best_score / 100.0
    if best_score < config.FUZZY_THRESHOLD:
        return "unknown", norm_score

    return best_cmd, norm_score


def match_command(transcript: str) -> dict:
    """
    Main entry point: map transcript → command label + confidence.

    Strategy:
      1. Try keyword matching first (fast, interpretable)
      2. If uncertain (score < 1.0), fall back to fuzzy match
      3. Return whichever method produced the stronger confident result

    Args:
        transcript: Hindi text string from Whisper.

    Returns:
        dict with keys:
          - "command": str (label or "unknown")
          - "confidence": float 0.0–1.0
          - "method": str ("keyword" | "fuzzy" | "unknown")
          - "transcript": str (echoed back)
    """
    if not transcript or not transcript.strip():
        return {
            "command": "unknown",
            "confidence": 0.0,
            "method": "unknown",
            "transcript": transcript,
        }

    kw_cmd, kw_score = _keyword_match(transcript)
    fz_cmd, fz_score = _fuzzy_match(transcript)

    # Prefer keyword match when it's confident (exact device + state hit)
    if kw_score >= 1.0:
        return {
            "command": kw_cmd,
            "confidence": kw_score,
            "method": "keyword",
            "transcript": transcript,
        }

    # Fall back to fuzzy match
    if fz_cmd != "unknown":
        return {
            "command": fz_cmd,
            "confidence": fz_score,
            "method": "fuzzy",
            "transcript": transcript,
        }

    # Partial keyword match is still better than nothing
    if kw_score > 0.0 and kw_cmd != "unknown":
        return {
            "command": kw_cmd,
            "confidence": kw_score * 0.5,   # penalize partial matches
            "method": "keyword_partial",
            "transcript": transcript,
        }

    return {
        "command": "unknown",
        "confidence": 0.0,
        "method": "unknown",
        "transcript": transcript,
    }


if __name__ == "__main__":
    # Quick smoke test
    test_phrases = [
        "लाइट चालू करो",
        "पंखा बंद कर दो",
        "AC खोलो",
        "टीवी बंद",
        "hello world",    # should be unknown
    ]
    for phrase in test_phrases:
        result = match_command(phrase)
        print(f"'{phrase}' → {result['command']} ({result['method']}, conf={result['confidence']:.2f})")
