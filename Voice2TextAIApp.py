import streamlit as st
from faster_whisper import WhisperModel
import os
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings,WebRtcMode
import numpy as np
import av

# Title and sidebar for the app
st.title("AI Podcast Transcriber with Audio Recording")
st.sidebar.header("Transcription Section")

# Caching the Whisper model
@st.cache_resource
def load_model():
    model_size = "tiny"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

# Ensure the transcript is stored in session_state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# Class to process live audio data
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffers = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.buffers.append(audio)
        return frame

# Audio recording button using webrtc_streamer
webrtc_ctx = webrtc_streamer(
    key="audio-record",
     mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    },
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

start_transcribe_button = st.sidebar.button("Start Transcription")

if start_transcribe_button:
    if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.buffers:
        st.sidebar.markdown("Note: AI is acting on it! Please wait...")

        model = load_model()

        st.sidebar.info("Transcribing recorded audio...")

        audio_data = np.concatenate(webrtc_ctx.audio_processor.buffers, axis=1)
        audio_bytes = BytesIO(audio_data.tobytes())

        # Display a progress bar during processing
        with st.spinner('Transcribing...'):
            segments, info = model.transcribe(audio_bytes, beam_size=5, language="en", condition_on_previous_text=False)

        # Show detected language and probability
        st.write(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

        transcript = ""

        for segment in segments:
            st.markdown(segment.text)
            transcript += f"{segment.text}"
            st.session_state.transcript += transcript

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

        # Option to display summary as bullet points
        if st.checkbox("Display summary in bullet points"):
            bullets = st.session_state.transcript.split(".")
            for bullet in bullets:
                if bullet.strip():
                    st.markdown(f"- {bullet.strip()}")
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