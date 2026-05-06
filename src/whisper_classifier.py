"""
src/whisper_classifier.py — Whisper encoder + Classification Head

Architecture:
  - Whisper-tiny encoder (pre-trained, frozen initially)
  - Global average pool over time
  - 2-layer MLP classification head (trainable)

This is the "fine-tuning" approach: we leverage Whisper's pre-trained
understanding of Hindi phonetics, and train only the head on 240 clips.
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_SIZE = "tiny"                  # 39M params encoder
HF_MODEL_ID  = f"openai/whisper-{WHISPER_SIZE}"
ENCODER_DIM  = 384                     # whisper-tiny encoder output dim
SAMPLE_RATE  = 16000
CLIP_DURATION = 4                      # seconds — must match features.py


class WhisperCommandClassifier(nn.Module):
    """
    Whisper encoder (frozen) → GAP → MLP head → 8-class softmax.

    Frozen encoder = transfer learning: we keep Whisper's speech knowledge
    and only train the small classification head on your 240 clips.
    """

    def __init__(self, num_classes: int = 8, unfreeze_last_n: int = 0):
        super().__init__()

        # Load pre-trained Whisper encoder only (no decoder needed)
        whisper = WhisperModel.from_pretrained(HF_MODEL_ID)
        self.encoder = whisper.encoder

        # Freeze all encoder params by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last N encoder blocks for deeper fine-tuning
        if unfreeze_last_n > 0:
            for block in self.encoder.layers[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Light classification head
        self.head = nn.Sequential(
            nn.LayerNorm(ENCODER_DIM),
            nn.Linear(ENCODER_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch, 80, time) mel-spectrogram from WhisperFeatureExtractor

        Returns:
            logits: (batch, num_classes)
        """
        enc_out = self.encoder(input_features).last_hidden_state  # (B, T', 384)
        pooled  = enc_out.mean(dim=1)                              # (B, 384) GAP
        return self.head(pooled)                                    # (B, num_classes)
