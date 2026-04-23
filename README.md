# Hindi Smart-Home Command Recognizer (T11.3)

**SMAI Assignment 3 · Team Project · Easiest Path: Whisper-tiny + String Match**

---

## What This Does

A Streamlit app that recognizes **8 Hindi smart-home voice commands**:

| Command | Hindi Phrase |
|---------|-------------|
| light_on  | लाइट चालू करो |
| light_off | लाइट बंद करो  |
| fan_on    | पंखा चालू करो  |
| fan_off   | पंखा बंद करो   |
| ac_on     | AC चालू करो   |
| ac_off    | AC बंद करो    |
| tv_on     | TV चालू करो   |
| tv_off    | TV बंद करो    |

**How it works (zero ML training):**
1. Whisper-tiny (39M params, CPU-capable) transcribes audio → Hindi text
2. Keyword scoring + fuzzy string matching maps transcript → command label
3. Streamlit UI shows transcript, command, and simulated action

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app (Whisper model downloads ~150 MB on first run)
streamlit run app.py
```

---

## Recording Your Own Clips (Required!)

Each team member records 8 commands × 10 takes:

```bash
python data/record_audio.py --student YOUR_NAME
```

Clips are saved to `data/self_recorded/<command>/`. The folder name = ground truth label.

---

## Evaluate Accuracy

```bash
python src/evaluate.py
# or specify a custom directory:
python src/evaluate.py --data_dir data/self_recorded
```

---

## Download IndicVoices Dataset

```bash
# First accept terms at https://huggingface.co/datasets/ai4bharat/IndicVoices
# Then log in:
huggingface-cli login

# Download (warning: large dataset, use --max_samples to limit)
python data/download_indicvoices.py --max_samples 1000
```

---

## Test Pipeline from Command Line

```bash
# Transcription only
python src/transcribe.py path/to/audio.wav

# Full pipeline
python src/pipeline.py path/to/audio.wav

# Matcher only (text input)
python src/matcher.py   # runs built-in smoke tests
```

---

## Project Structure

```
hindi-command-recognizer/
├── app.py                        # Streamlit app
├── config.py                     # Commands, keywords, paths — edit here
├── requirements.txt
├── src/
│   ├── transcribe.py             # Whisper-tiny wrapper
│   ├── matcher.py                # Keyword + fuzzy matching
│   ├── pipeline.py               # End-to-end predict()
│   └── evaluate.py               # Accuracy evaluation
└── data/
    ├── record_audio.py           # CLI recording helper
    ├── download_indicvoices.py   # HF dataset downloader
    ├── self_recorded/            # YOUR recordings go here
    │   ├── light_on/
    │   ├── light_off/ …
    └── indicvoices_hindi/        # Downloaded dataset (gitignored)
```

---

## Team Action Items

| # | Action | Who |
|---|--------|-----|
| 1 | Accept IndicVoices terms on HuggingFace | All 3 members |
| 2 | `huggingface-cli login` with shared token | One member |
| 3 | Record 8 commands × 10 takes each | **All 3 members** |
| 4 | `pip install -r requirements.txt` | All members |
| 5 | Run `streamlit run app.py` to demo | All members |
