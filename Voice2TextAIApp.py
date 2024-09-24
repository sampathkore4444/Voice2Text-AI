import streamlit as st
from faster_whisper import WhisperModel
import os
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
import numpy as np

# Title and sidebar for the app
st.title("Voice2Text AI Transcriber")
st.sidebar.header("Transcription Section")

# Record audio using the audio recorder
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41000)
st.audio(audio_bytes, format="audio/wav")

# Caching the Whisper model
@st.cache_resource
def load_model():
    model_size = "tiny"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

# Ensure the transcript is stored in session_state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

start_transcribe_button = st.sidebar.button("Start Transcription")

if start_transcribe_button:
    if audio_bytes:
        

        st.sidebar.markdown("Note: AI is acting on it! Please wait...")

        model = load_model()

        st.sidebar.info("Transcribing recorded audio...")

        # Display a progress bar during processing
        with st.spinner('Transcribing...'):
            segments, info = model.transcribe(BytesIO(audio_bytes), beam_size=5, condition_on_previous_text=False)

        # Show detected language and probability
        st.write(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

        transcript = ""

        for segment in segments:
            # st.markdown(segment.text)
            transcript += f"{segment.text} "
        
        st.session_state.transcript = transcript

        st.text_area("Transcript", transcript, height=300)

        # Allow users to download the transcript
        if st.session_state.transcript:
            st.download_button(
                label="Download Transcript",
                data=st.session_state.transcript,
                file_name='recorded_audio_transcript.txt',
                mime='text/plain'
            )

        st.sidebar.success("Transcription Complete")

       
    else:
        st.sidebar.error("Please record some audio first.")

# Footer styling
st.markdown("""
<style>
body {
    background-color: #f2f2f2;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    color: #3366cc;
}
button {
    background-color: #ff6600 !important;
    color: white !important;
}
.stButton>button {
    background-color: #ff6600 !important;
    color: white !important;
    border-radius: 10px;
    width: 100%;
}
.stAudio {
    border: 2px solid #ff6600 !important;
    padding: 10px;
    background-color: #fff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("----")
