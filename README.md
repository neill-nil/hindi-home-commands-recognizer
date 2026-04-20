# Indic Speech Command Recognizer (T11.3 Phase 1)

This project contains the backend testing script and basic Streamlit implementation for translating raw audio into mapped smart-home commands in Hindi/Hinglish (e.g. `LIGHT_ON`, `FAN_OFF`) via `openai/whisper-tiny`.

## Setup

1. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Test

### 1. Test the Backend Directly (Recommended first step)
Before connecting the UI, test if Whisper accurately detects your mic's `.wav` file structure:
1. Record a quick audio file on your phone saying a command (e.g., "batti jalao").
2. Transfer that file to this same folder and name it `test.wav`.
3. Run the backend script:
   ```bash
   python backend_test.py
   ```
4. Look at the terminal output to see the exact transcript and whether it successfully mapped to an Intent.

### 2. Run the Streamlit Interface
Once `backend_test.py` proves Whisper is working, launch the UI to record straight from your browser:
```bash
streamlit run app.py
```
*Note: Make sure to click "Allow" when the browser asks for microphone permissions.*

---

## Modifying the Hindi/Hinglish Commands

We are currently equipped for **BOTH** Devanagari and Hinglish depending on how Whisper handles your speech. 

**To update the 8 chosen commands and their keywords:**
1. Open `backend_test.py`
2. Scroll to the `COMMAND_INTENTS` list at the very top.
3. Modify the list! The `intent` is the backend action name, and `keywords` is a list of ALL acceptable strings.

**Example Modification:**
```python
# In backend_test.py
COMMAND_INTENTS = [
    {
        "intent": "DOOR_OPEN",
        "keywords": ["darwaza kholo", "door open", "kholo", "दरवाज़ा खोलो"] 
    }
]
```

> **Why both Hinglish and Devanagari?**
> The `whisper-tiny` pipeline will typically transcribe native Hindi to Devanagari (e.g. "बत्ती"). However, sometimes with accents it gets predicted as English phonetic (Hinglish/Romanized). Include both in your `keywords` list to guarantee it matches your intent!
