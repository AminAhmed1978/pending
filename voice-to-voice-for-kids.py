import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from gtts import gTTS
import streamlit_webrtc as webrtc
import whisper
import tempfile
import os
import io
import numpy as np
from pydub import AudioSegment

# Load the models
whisper_model = whisper.load_model("base")
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Set the source and target language to Urdu
source_lang = "ur"
target_lang = "ur"

# Streamlit layout
st.title("Voice-to-Voice Chatbot for Kids - Urdu")

# WebRTC-based real-time audio capture
def process_audio(audio_bytes):
    # Save the audio in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        audio_path = tmp_file.name

    # Load audio and perform transcription using Whisper
    audio = whisper.load_audio(audio_path)
    result = whisper_model.transcribe(audio, language='ur')  # Transcribe in Urdu
    st.write("You said (in Urdu):", result["text"])

    # Generate response using mBART model
    inputs = tokenizer(result["text"], return_tensors="pt")
    translated_tokens = mbart_model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    st.write("Chatbot's response:", translated_text)

    # Convert the response text to speech (gTTS)
    tts = gTTS(translated_text, lang=target_lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tts.save(tmp_audio.name)
        st.audio(tmp_audio.name)

# Real-time audio input from microphone
webrtc_stream = webrtc.WebRtcMode.SENDONLY
ctx = webrtc.StreamerContext(
    mode=webrtc_stream, 
    key="key"
)

# Button to process the real-time voice input
if st.button("Record and Process Voice"):
    if ctx.audio_frame:
        audio_frame = ctx.audio_frame.to_ndarray()
        # Convert the audio_frame to bytes for processing
        audio_bytes = io.BytesIO()
        audio = AudioSegment(audio_frame.tobytes(), sample_width=2, frame_rate=16000, channels=1)
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)
        process_audio(audio_bytes.read())

