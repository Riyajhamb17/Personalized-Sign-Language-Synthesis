import streamlit as st
import subprocess
import os
import tempfile
import speech_recognition as sr
import whisper

# -----------------------------------------
# Whisper model (local)
# -----------------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")   
whisper_model = load_whisper()


# -----------------------------------------
# STT: Audio Upload  → Whisper
# -----------------------------------------
def speech_to_text_file(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


# -----------------------------------------
# STT: Microphone → Google STT
# -----------------------------------------
def speech_to_text_mic(recorded_audio):
    recognizer = sr.Recognizer()

    # Convert Streamlit audio to WAV temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(recorded_audio.getvalue())
        temp_path = tmp.name

    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except:
        return ""


# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.set_page_config(layout="wide", page_title="SignAI Demo")

st.markdown("""
<div style='text-align:center; font-size:32px; font-weight:700;'>SignAI — Text to Sign Language</div>
<div style='text-align:center; font-size:16px; color:#888;'>Text, Upload Audio, or Use Microphone → Auto Sign Animation</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])


# ==============================
# LEFT: INPUT AREA
# ==============================
with col1:
    st.header("Input Method")

    option = st.selectbox(
        "Choose input type:",
        ["Text Input", "Audio File Upload", "Microphone Recording"]
    )

    user_text = None

    # -----------------------------
    # TEXT INPUT
    # -----------------------------
    if option == "Text Input":
        user_text = st.text_area("Enter a sentence:", height=150)

    # -----------------------------
    # AUDIO FILE UPLOAD
    # -----------------------------
    elif option == "Audio File Upload":
        uploaded = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3"])

        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                audio_path = tmp.name

            st.success(f"Uploaded: {uploaded.name}")

            with st.spinner("Transcribing audio using Whisper..."):
                user_text = speech_to_text_file(audio_path)

            st.text_area("Recognized Text:", value=user_text, height=150)

    # -----------------------------
    # MICROPHONE RECORDING
    # -----------------------------
    elif option == "Microphone Recording":
        recorded = st.audio_input("Record using microphone")

        if recorded:
            st.success("Recording captured!")

            with st.spinner("Converting microphone audio to text..."):
                user_text = speech_to_text_mic(recorded)

            st.text_area("Recognized Text:", value=user_text, height=150)


    # -----------------------------
    # GENERATE BUTTON
    # -----------------------------
    if st.button("Generate Sign Animation"):
        if not user_text or user_text.strip() == "":
            st.error("Please enter or record some text first.")
        else:
            st.info("Generating sign animation...")

            cmd = f'python retriever_frontend.py --text "{user_text}"'

            with st.spinner("Processing... This may take a moment."):
                subprocess.run(cmd, shell=True)

            st.success("Animation Ready!")


# ==============================
# RIGHT: OUTPUT AREA
# ==============================
with col2:
    st.header("Sign Animation Output")

    if os.path.exists("sign_output.mp4"):
        st.video("sign_output.mp4", format="video/mp4", start_time=0, width=600)
    else:
        st.info("Your generated sign animation will appear here.")
