import os
import torch
from transformers import pipeline

# Load the model outside the prediction function so it stays in memory in Streamlit later
print("Loading Whisper model... (this may take a moment on first run)")
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    # We let the model auto-detect language for now if you want a mix of Hindi and Hinglish. 
    # To strictly force Hindi transcription: generate_kwargs={"language": "hindi"}
    # To force translation to English: generate_kwargs={"task": "translate"}
)

# ==============================================================================
# 🛠️ HOW TO MODIFY COMMANDS:
# Add or remove your exact 8 commands here. 
# You can include BOTH Hinglish (romanized) and Hindi (Devanagari) variations
# depending on what Whisper ends up spitting out during your real-world tests.
# ==============================================================================
COMMAND_INTENTS = [
    {
        "intent": "LIGHT_ON",
        "keywords": ["batti jalao", "light on", "turn on light", "बत्ती जलाओ"]
    },
    {
        "intent": "LIGHT_OFF",
        "keywords": ["batti bujhao", "batti band", "light off", "turn off light", "बत्ती बुझाओ", "बत्ती बंद"]
    },
    {
        "intent": "FAN_ON",
        "keywords": ["pankha chalao", "fan on", "pankha on", "पंखा चलाओ"]
    },
    {
        "intent": "FAN_OFF",
        "keywords": ["pankha band karo", "fan off", "pankha off", "पंखा बंद करो"]
    },
    {
        "intent": "TV_ON",
        "keywords": ["tv chalu karo", "tv on", "television on", "टीवी चालू करो"]
    },
    {
        "intent": "TV_OFF",
        "keywords": ["tv band karo", "tv off", "television off", "टीवी बंद करो"]
    },
    {
        "intent": "AC_ON",
        "keywords": ["ac chalu karo", "ac on", "एसी चालू करो"]
    },
    {
        "intent": "AC_OFF",
        "keywords": ["ac band karo", "ac off", "एसी बंद करो"]
    }
]

def determine_intent(transcript: str) -> str:
    """Takes a transcript string and returns the matching intent."""
    text = transcript.lower().strip()
    
    for command in COMMAND_INTENTS:
        for keyword in command["keywords"]:
            # Simple substring matching
            if keyword in text:
                return command["intent"]
                
    return "UNKNOWN_COMMAND"

def process_audio(audio_input) -> tuple[str, str]:
    """
    Takes an audio file path or raw bytes, passes it to Whisper, 
    and returns the (transcript, intent).
    """
    # Pass the audio into the pipeline
    result = whisper_pipeline(audio_input)
    transcript = result["text"]
    
    # Process the transcript against our known intents
    intent = determine_intent(transcript)
    
    return transcript, intent

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 🧪 HOW TO TEST LOCALLY WITHOUT STREAMLIT
    # 1. Record a quick voice note saying "batti jalao" on your phone.
    # 2. Save it to your project folder as "test.wav".
    # 3. Change the path below and run: python backend_test.py
    # ---------------------------------------------------------
    
    TEST_AUDIO_FILE = "test.wav"
    
    if os.path.exists(TEST_AUDIO_FILE):
        print(f"Testing with {TEST_AUDIO_FILE}...")
        try:
            transcript, intent = process_audio(TEST_AUDIO_FILE)
            print(f"\n📝 Raw Transcript: {transcript}")
            print(f"🤖 Detected Action: {intent}\n")
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
    else:
        print(f"\n⚠️ Could not find test file: '{TEST_AUDIO_FILE}'")
        print("Please place an audio file named 'test.wav' in the same folder and run again.")
