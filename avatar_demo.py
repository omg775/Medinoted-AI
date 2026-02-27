import streamlit as st
import time
import os
import io
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

# --- Constants & Configuration ---
HEALTH_KEYWORDS = ["doctor", "hospital", "medicine", "disease", "health", "sick", "pain", "treatment", "virus", "fever", "pharmacy", "medical"]

def load_css(file_path="avatar.css"):
    try:
        with open(file_path, "r") as f:
            return f"<style>{f.read()}</style>"
    except FileNotFoundError:
        return ""

def get_avatar_html(is_talking=False):
    talking_class = "is-talking" if is_talking else ""
    return f"""
    <div class="scene">
        <div class="avatar {talking_class}">
            <div class="face front">
                <div class="quadrant q-red"></div>
                <div class="quadrant q-green"></div>
                <div class="quadrant q-blue"></div>
                <div class="quadrant q-yellow"></div>
                <div class="eyes">
                    <div class="eye"><div class="pupil"></div></div>
                    <div class="eye"><div class="pupil"></div></div>
                </div>
                <div class="mouth-container">
                    <div class="mouth"></div>
                </div>
            </div>
            <div class="face back">
                <div class="quadrant q-red"></div><div class="quadrant q-green"></div>
                <div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div>
            </div>
            <div class="face right">
                <div class="quadrant q-red"></div><div class="quadrant q-green"></div>
                <div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div>
            </div>
            <div class="face left">
                <div class="quadrant q-red"></div><div class="quadrant q-green"></div>
                <div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div>
            </div>
            <div class="face top">
                <div class="quadrant q-red"></div><div class="quadrant q-green"></div>
                <div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div>
            </div>
            <div class="face bottom">
                <div class="quadrant q-red"></div><div class="quadrant q-green"></div>
                <div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div>
            </div>
        </div>
    </div>
    """

def is_health_topic(text):
    text = text.lower()
    return any(keyword in text for keyword in HEALTH_KEYWORDS)

# --- App Layout ---
st.set_page_config(page_title="Microsoft 3D Avatar Assistant", layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.title("Settings")
    # Try to get API key from secrets first
    default_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")

st.title("Microsoft 3D Care Assistant")
st.write("Hello! I am your interactive Microsoft 3D Assistant. Speak to me by clicking the microphone below.")

col_avatar, col_chat = st.columns([1, 1])

if "talking" not in st.session_state:
    st.session_state.talking = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with col_avatar:
    avatar_placeholder = st.empty()
    avatar_placeholder.markdown(get_avatar_html(st.session_state.talking), unsafe_allow_html=True)

with col_chat:
    st.subheader("Voice Interaction")
    audio = mic_recorder(
        start_prompt="Speak with Assistant 🎤",
        stop_prompt="Stop Recording ⏹️",
        key='recorder'
    )

    if audio:
        if not api_key:
            st.warning("Please provide an OpenAI API Key in the sidebar.")
        else:
            client = OpenAI(api_key=api_key)
            
            with st.spinner("Listening..."):
                # 1. Transcribe (STT)
                audio_bio = io.BytesIO(audio['bytes'])
                audio_bio.name = "audio.wav"
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_bio
                ).text
                
                st.info(f"You: {transcript}")
                
                # 2. Safety Check
                if is_health_topic(transcript):
                    st.error("🚫 Security Protocol: I am programmed to avoid clinical health topics for safety. Let's talk about something else!")
                else:
                    # 3. AI Response
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a friendly, helpful Microsoft 3D Assistant. Be concise (max 2 sentences)."},
                            {"role": "user", "content": transcript}
                        ]
                    ).choices[0].message.content
                    
                    # 4. Generate Voice (TTS)
                    audio_resp = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=response
                    )
                    
                    # 5. Play and Animate
                    st.audio(audio_resp.content, format="audio/mp3", autoplay=True)
                    
                    # Toggle talking animation
                    st.session_state.talking = True
                    avatar_placeholder.markdown(get_avatar_html(True), unsafe_allow_html=True)
                    
                    # Display response
                    st.success(f"Assistant: {response}")
                    
                    # Estimate duration and reset animation
                    # Average speech speed is ~150 words per minute -> ~2.5 words per second
                    # Estimating based on char count for better granularity
                    sleep_duration = max(2, len(response) * 0.05)
                    time.sleep(sleep_duration)
                    
                    st.session_state.talking = False
                    avatar_placeholder.markdown(get_avatar_html(False), unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Microsoft Brand Aesthetics | Powered by OpenAI")

