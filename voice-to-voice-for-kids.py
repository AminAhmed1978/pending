import whisper
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer  # mBART imports
from gtts import gTTS
import os
import tempfile
import streamlit as st

# Load models
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def transcribe_audio(audio_file):
    st.write("Transcribing audio...")
    # Use Whisper model to transcribe speech to text
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(audio_file)
    return result['text']

def generate_response(input_text, source_lang, target_lang):
    # Tokenize input
    tokenizer.src_lang = source_lang
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    # Generate translation
    generated_ids = model.generate(input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

def text_to_speech(text, lang='ur'):
    # Convert text to speech using Google Text-to-Speech
    tts = gTTS(text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

def main():
    st.title("Voice-to-Voice Chatbot for Kids in Urdu")
    st.write("Ask anything in Urdu, and get a response in Urdu!")
    
    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        # Transcribe audio to text
        transcribed_text = transcribe_audio(audio_file)
        st.write(f"Transcribed Text: {transcribed_text}")
        
        # Generate response
        response_text = generate_response(transcribed_text, "ur_UR", "ur_UR")
        st.write(f"Response Text: {response_text}")
        
        # Convert response text to speech
        response_audio = text_to_speech(response_text)
        st.audio(response_audio, format='audio/mp3')

if __name__ == "__main__":
    main()
