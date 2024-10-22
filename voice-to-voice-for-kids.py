import whisper # Import the whisper module
import streamlit as st
import tempfile
import os
from gtts import gTTS  # Import gTTS for text-to-speech
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast  # Import mBART models

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Load mBART model and tokenizer from Hugging Face
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Set the source and target language codes for Urdu
source_lang = "ur_IN"
target_lang = "ur_IN"
tokenizer.src_lang = source_lang

# Streamlit UI setup
st.title("Voice-to-Voice Chatbot for Kids (in Urdu)")

# File uploader for audio input (wav, mp3, or m4a format)
audio_file = st.file_uploader("Upload an audio file (Urdu)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save the uploaded audio temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(audio_file.read())

    # Transcribe audio to text (Urdu transcription)
    transcription = whisper_model.transcribe(temp_file.name, language="ur")
    st.write("Transcribed Text: ", transcription["text"])

    # Tokenize the transcription and generate a response using mBART
    input_tokens = tokenizer(transcription["text"], return_tensors="pt")
    generated_tokens = mbart_model.generate(**input_tokens)
    response_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    st.write("Chatbot Response: ", response_text)

    # Convert chatbot response back to speech using gTTS (Urdu language)
    tts = gTTS(text=response_text, lang='ur')
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)

    # Play the response audio
    audio_file_path = tts_file.name
    audio_bytes = open(audio_file_path, "rb").read()
    st.audio(audio_bytes, format='audio/mp3')

    # Clean up temporary files
    temp_file.close()
    os.unlink(temp_file.name)
    os.unlink(tts_file.name)
