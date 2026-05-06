"""
src/finetune_whisper.py — Fine-tune Whisper-tiny on our datasets

This script leverages Hugging Face Transformers to load the pre-trained openai/whisper-tiny,
process our self-recorded and mined Hindi datasets, and fine-tune it for improved
intent recognition on our specific smart-home commands.
"""

import os
import sys
import json
import librosa
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Temporarily remove local src dir from sys.path so we import HuggingFace 'evaluate', not 'src/evaluate.py'
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
import evaluate
sys.path.insert(0, _script_dir)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        audio_path = item["audio_path"]
        transcript = item["transcript"]

        # Load audio
        y, _ = librosa.load(audio_path, sr=16000)
        
        # Extract features
        input_features = self.processor.feature_extractor(y, sampling_rate=16000).input_features[0]

        # Tokenize labels
        labels = self.processor.tokenizer(transcript).input_ids

        return {"input_features": input_features, "labels": labels}

def prepare_dataset():
    data_list = []
    
    # 1. Load self_recorded
    self_recorded_dir = Path(config.SELF_RECORDED_DIR)
    if self_recorded_dir.exists():
        for label_dir in self_recorded_dir.iterdir():
            if not label_dir.is_dir():
                continue
            cmd = label_dir.name
            if cmd not in config.CANONICAL_PHRASES:
                continue
            phrase = config.CANONICAL_PHRASES[cmd]
            
            for audio_file in label_dir.glob("*.wav"):
                data_list.append({
                    "audio_path": str(audio_file),
                    "transcript": phrase
                })

    # 2. Load mined_local_fleurs
    mined_dir = Path(config.MINED_LOCAL_FLEURS_DIR)
    manifest_path = mined_dir / "mined_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            for item in manifest:
                audio_path = mined_dir / item["file"]
                if audio_path.exists():
                    data_list.append({
                        "audio_path": str(audio_path),
                        "transcript": item["transcript"]
                    })
    
    return data_list

def main():
    print("="*60)
    print("       Fine-Tuning Whisper ASR (Hugging Face)")
    print("="*60)
    
    data_list = prepare_dataset()
    if not data_list:
        print("No training data found!")
        sys.exit(1)
        
    print(f"Loaded {len(data_list)} audio samples for fine-tuning.")
    
    model_id = "openai/whisper-tiny"
    print(f"Loading {model_id} processor and model...")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language="hi", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_id, language="hi", task="transcribe")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Freeze the encoder to speed up training and save memory
    model.freeze_encoder()
    
    dataset = WhisperDataset(data_list, processor)
    
    # Train/Eval split (80/20)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=generator)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    output_dir = os.path.join(config.BASE_DIR, "models", "whisper_finetuned")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=25,
        max_steps=300,  # Full run: ~5–8 mins on Mac. Change to 50 for a quick test.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,  # True if CUDA, Mac MPS usually prefers False
        eval_strategy="steps",  # replaces deprecated evaluation_strategy
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    
    # Simple WER metric
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
