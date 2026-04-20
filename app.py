import streamlit as st
import tempfile
import os
from streamlit_mic_recorder import mic_recorder
from backend_test import process_audio

st.set_page_config(page_title="Indic Voice Commands", page_icon="🎙️")

st.title("🎙️ Indic Smart-Home Voice Commands")
st.write("Hold the mic button to record your command.")
st.markdown("Try saying: *'batti jalao'*, *'pankha band karo'*, or their exact Hindi counterparts.")

# Use the mic_recorder component to capture audio directly from the browser
audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=False,
    use_container_width=False,
    format="wav",
    key="mic_recorder"
)

if audio:
    st.info("Got audio! Processing with Whisper...")
    
    # mic_recorder returns a dict containing an audio bytes object
    audio_bytes = audio["bytes"]
    
    # Save the bytes to a temporary wav file for the Whisper pipeline
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Run the backend inference function
        with st.spinner("Transcribing..."):
            transcript, intent = process_audio(tmp_path)
            
        st.subheader("Transcription:")
        st.write(f"> {transcript}")
        
        st.subheader("Detected Action:")
        if intent != "UNKNOWN_COMMAND":
            st.success(f"**⚡ {intent}**")
        else:
            st.error(f"**❓ {intent}**")
            st.caption("Command not recognized. If it should have been, check `backend_test.py` and add the required keywords to the `COMMAND_INTENTS` list!")
            
    except Exception as e:
        st.error(f"Error processing audio: {e}")
    finally:
        # Cleanup temporary file to avoid clutter
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
