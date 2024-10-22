import whisper
import requests
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
import streamlit as st

# Load Whisper model for transcription
model = whisper.load_model("base")

# Groq API details
groq_api_url = "https://api.groq.com/v1/your-endpoint"  # Replace with the actual Groq API endpoint
groq_api_key = "gsk_3U90LE9QszpPzMGIeDUYWGdyb3FYVTj75zH6gcWo7I4Ym28FU8gmY"

# Set up Microsoft Azure for TTS
speech_config = SpeechConfig(subscription="your_azure_subscription_key", region="your_azure_region")
audio_config = AudioConfig(use_default_speaker=True)
speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

def get_groq_response(prompt):
    """Generate a response using Groq API."""
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }
    
    response = requests.post(groq_api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("response_text", "")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "Sorry, I couldn't generate a response."

# Streamlit UI for the chatbot
st.title("Urdu Voice-to-Voice Chatbot for Kids")
st.write("Ask your question in Urdu by recording your voice.")

# Voice input from the user
audio_input = st.file_uploader("Upload or record your voice in Urdu", type=["wav", "mp3"])

if audio_input is not None:
    # Transcribe the voice input using Whisper
    transcription = model.transcribe(audio_input)["text"]
    st.write(f"Question in Urdu: {transcription}")

    # Generate a response using Groq API
    response = get_groq_response(f"Respond in Urdu to the question: {transcription}")
    st.write(f"Response in Urdu: {response}")

    # Convert response to speech using Azure TTS
    result = speech_synthesizer.speak_text_async(response).get()
    
    # Play the generated audio response
    st.audio(result.audio_data, format="audio/wav")
