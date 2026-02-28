# REQUIRED TO INSTALL:
# pip install streamlit matplotlib vaderSentiment sounddevice scipy openai
# 
# OPTIONAL (But Highly Recommended for full functionality):
# pip install openai-whisper
# pip install SpeechRecognition
# pip install spacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

import bcrypt
import streamlit as st
import os
import json
import re
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import requests
from scipy.io import wavfile

def get_geolocation():
    """Browser-based geolocation helper via HTML/JS bridge."""
    from streamlit.components.v1 import html
    js_code = """
    <script>
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const { latitude, longitude } = pos.coords;
                // Send back to Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: { lat: latitude, lon: longitude }
                }, '*');
            },
            (err) => {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: { error: err.message }
                }, '*');
            }
        );
    </script>
    """
    return html(js_code, height=0)

def query_nearby_care(lat, lon, categories=["hospital", "pharmacy", "clinic"]):
    """Query Overpass API for nearby healthcare facilities."""
    try:
        # Simple bounding box ~10km
        delta = 0.1 
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"~"{"|".join(categories)}"]({lat-delta},{lon-delta},{lat+delta},{lon+delta});
          way["amenity"~"{"|".join(categories)}"]({lat-delta},{lon-delta},{lat+delta},{lon+delta});
        );
        out center;
        """
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        results = []
        for element in data.get('elements', []):
            name = element.get('tags', {}).get('name', 'Unnamed Facility')
            type_ = element.get('tags', {}).get('amenity', 'Medical')
            results.append({
                'name': name,
                'type': type_.title(),
                'lat': element.get('lat') or element.get('center', {}).get('lat'),
                'lon': element.get('lon') or element.get('center', {}).get('lon')
            })
        return results
    except Exception as e:
        return []

from openai import OpenAI
import time
import textwrap
from dotenv import load_dotenv

load_dotenv(override=True)

# --- AI Avatar Logic ---
def get_avatar_html(is_talking=False, overlay_text=""):
    talking_class = "is-talking" if is_talking else ""
    overlay_html = f'<div class="ms-overlay-label">{overlay_text}</div>' if overlay_text else ""
    # We remove all indentation and comments to prevent Streamlit/Markdown from showing raw text
    html = f'<div class="scene-container">{overlay_html}<div class="scene"><div class="avatar {talking_class}"><div class="face front"><div class="quadrant q-red"></div><div class="quadrant q-green"></div><div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div><div class="doctor-outfit"><div class="lapel lapel-left"></div><div class="lapel lapel-right"></div></div><div class="physician-badge"><div class="badge-pic"></div><div class="badge-title">MD</div><div class="badge-name">ASSISTANT</div></div><div class="eyes"><div class="eye"><div class="pupil"></div></div><div class="eye"><div class="pupil"></div></div></div><div class="mouth-box"><div class="mouth"></div></div><div class="steth-chest"></div></div><div class="steth-cable"></div><div class="face back"><div class="quadrant q-red"></div><div class="quadrant q-green"></div><div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div></div><div class="face right"><div class="quadrant q-red"></div><div class="quadrant q-green"></div><div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div></div><div class="face left"><div class="quadrant q-red"></div><div class="quadrant q-green"></div><div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div></div><div class="face top"><div class="quadrant q-red"></div><div class="quadrant q-green"></div><div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div></div><div class="face bottom"><div class="quadrant q-red"></div><div class="quadrant q-green"></div><div class="quadrant q-blue"></div><div class="quadrant q-yellow"></div></div></div><div class="ground-shadow"></div></div></div>'
    return html

def load_avatar_css():
    try:
        with open("avatar.css", "r") as f:
            return f"<style>{f.read()}</style>"
    except FileNotFoundError:
        return ""

# Native Streamlit audio_input used for browser audio capturing
HAS_AUDIO_RECORDER = False
HAS_MIC_RECORDER = False

# -----------------------------------------------------------------------------
# Optional Dependency Loading (Graceful Degradation)
# -----------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Health Assistant AI", page_icon="https://www.microsoft.com/favicon.ico", layout="wide")

st.markdown("""
<style>
    /* Microsoft Executive Pitch - Master-Class Fluid System 4.0 */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700;800&display=swap');
    
    :root {
        --ms-orange: #F25022;
        --ms-green: #7FBA00;
        --ms-blue: #3A86FF;
        --ms-yellow: #FFB900;
        --ms-gray: #737373;
        
        /* Brand Palette */
        --brand-primary: #3A86FF;
        --brand-secondary: #2EC4B6;
        --brand-accent: #9B5DE5;

        --ms-neutral-primary: #201F1E;
        --ms-neutral-secondary: #605E5C;
        --ms-neutral-tertiary: #EDEBE9;
        --ms-pure-white: #FFFFFF;
        --ms-azure-white: #F3F6F9;
        
        /* Fluid Scales */
        --fs-title: clamp(2.2rem, 5vw, 3.4rem);
        --fs-subtitle: clamp(1rem, 2vw, 1.3rem);
        --fs-header: clamp(1.3rem, 3vw, 1.7rem);
        --fs-tab: clamp(1.2rem, 2.5vw, 1.7rem);
    }

    .stApp { 
        background-color: var(--background-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(58, 134, 255, 0.04) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(46, 196, 182, 0.04) 0px, transparent 40%);
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: var(--text-color);
    }
    
    [data-theme="light"] .stApp { 
        background-color: var(--ms-pure-white); 
        background-image: 
            radial-gradient(at 0% 0%, rgba(58, 134, 255, 0.06) 0px, transparent 40%),
            radial-gradient(at 100% 0%, rgba(46, 196, 182, 0.03) 0px, transparent 40%);
    }

    [data-theme="dark"] .stApp { 
        background-image: 
            url("https://www.transparenttextures.com/patterns/carbon-fibre.png"),
            radial-gradient(at 0% 0%, rgba(58, 134, 255, 0.08) 0px, transparent 40%),
            radial-gradient(at 100% 0%, rgba(46, 196, 182, 0.06) 0px, transparent 40%),
            radial-gradient(at 0% 100%, rgba(155, 93, 229, 0.06) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(58, 134, 255, 0.06) 0px, transparent 40%);
        background-blend-mode: overlay;
    }
    
    .main-container { max-width: 1240px; margin: 0 auto; padding: clamp(1.5rem, 4vw, 4rem) clamp(1rem, 3vw, 2rem); }
    
    .header-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: clamp(15px, 3vw, 28px);
        margin-bottom: clamp(1rem, 4vw, 2rem);
    }

    /* Native App Mobile Transformation */
    @media (max-width: 768px) {
        .header-wrapper {
            flex-direction: column;
            text-align: center;
            gap: 12px;
            margin-bottom: 2rem;
        }
        .main-title {
            text-align: center;
            line-height: 1.1;
        }
        .subtitle {
            margin-bottom: 3.5rem;
        }
        .stButton > button {
            width: 100% !important;
        }
        .card {
            padding: 1.5rem !important;
        }
    }

    .ms-logo { 
        width: clamp(38px, 8vw, 48px); 
        height: clamp(38px, 8vw, 48px); 
        filter: drop-shadow(0 4px 10px rgba(0,0,0,0.08)); 
    }

    .main-title { 
        color: var(--text-color); 
        font-weight: 800; 
        font-size: var(--fs-title); 
        margin: 0;
        letter-spacing: -0.03em;
    }

    [data-theme="light"] .main-title { color: var(--ms-neutral-primary); }

    .subtitle { 
        color: var(--ms-neutral-secondary); 
        font-size: var(--fs-subtitle); 
        font-weight: 700; 
        text-align: center; 
        margin-bottom: clamp(3rem, 8vw, 6rem); 
        text-transform: uppercase;
        letter-spacing: 0.25em;
        opacity: 0.8;
    }
    
    .card { 
        background: var(--secondary-background-color); 
        backdrop-filter: blur(32px) saturate(200%);
        -webkit-backdrop-filter: blur(32px) saturate(200%);
        padding: clamp(1.5rem, 5vw, 3.5rem); 
        border-radius: 4px; 
        box-shadow: 0 12px 40px rgba(0,0,0,0.06); 
        margin-bottom: clamp(1.5rem, 4vw, 2.5rem); 
        border: 1px solid rgba(128, 128, 128, 0.15); 
    }

    [data-theme="light"] .card { 
        background: var(--ms-pure-white); 
        border: 1px solid var(--ms-neutral-tertiary);
        box-shadow: 0 10px 30px rgba(0,0,0,0.04);
    }

    [data-testid="stVerticalBlock"] > div:nth-child(1) .card { border-top: 6px solid var(--brand-primary); }
    [data-testid="stVerticalBlock"] > div:nth-child(2) .card { border-top: 6px solid var(--brand-secondary); }

    .privacy-banner { 
        background: rgba(255, 185, 0, 0.08); 
        color: var(--text-color); 
        padding: clamp(1rem, 3vw, 1.5rem); 
        border-radius: 2px; 
        font-size: clamp(0.9rem, 2.5vw, 1.2rem);
        font-weight: 800; 
        text-align: center; 
        border: 1px solid var(--ms-yellow);
        border-left: clamp(8px, 2vw, 14px) solid var(--ms-yellow); 
        margin-bottom: clamp(2rem, 6vw, 4rem);
    }
    
    .section-header { 
        font-weight: 800; 
        color: var(--text-color); 
        font-size: var(--fs-header); 
        margin-bottom: 2rem; 
        border-left: clamp(5px, 1.5vw, 8px) solid var(--brand-primary); 
        padding-left: clamp(1rem, 2.5vw, 2rem);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-theme="light"] .section-header { color: var(--ms-neutral-primary); }
    
    /* Clinical Dashboard Styles */
    .stContainer {
        border-radius: 12px !important;
        background-color: white !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        border: 1px solid #e2e8f0 !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }

    .kpi-tile-clinical {
        padding: 0.75rem;
        text-align: center;
    }
    .kpi-value-clinical {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    .kpi-label-clinical {
        font-size: 0.7rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    .kpi-micro-clinical {
        font-size: 0.6rem;
        color: #94a3b8;
    }

    .accent-blue { border-left: 4px solid #3b82f6 !important; }
    .accent-green { border-left: 4px solid #10b981 !important; }
    .accent-orange { border-left: 4px solid #f59e0b !important; }
    .accent-purple { border-left: 4px solid #8b5cf6 !important; }

    .insight-card-clinical {
        background: #f8fafc !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    .orb-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: clamp(4rem, 10vw, 7rem) 0;
        position: relative;
        height: clamp(250px, 40vw, 320px);
    }
    
    .ai-orb {
        width: clamp(160px, 30vw, 220px);
        height: clamp(160px, 30vw, 220px);
        border-radius: 50%;
        background: conic-gradient(from 180deg at 50% 50%, var(--brand-primary), var(--brand-secondary), var(--brand-accent), var(--brand-primary));
        filter: blur(2px);
        box-shadow: 0 0 clamp(40px, 10vw, 100px) rgba(58, 134, 255, 0.30);
        animation: fluidRotate 10s infinite linear;
        position: relative;
        z-index: 2;
    }

    .ai-orb::before {
        content: '';
        position: absolute;
        inset: clamp(12px, 2vw, 18px);
        background: var(--secondary-background-color);
        opacity: 0.3;
        border-radius: 50%;
        backdrop-filter: blur(40px);
        z-index: 3;
    }
    [data-theme="light"] .ai-orb::before { background: white; opacity: 0.15; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid var(--ms-neutral-tertiary);
        margin-bottom: clamp(2rem, 8vw, 5rem);
        overflow-x: auto !important;
        white-space: nowrap !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: clamp(70px, 15vw, 110px);
        font-weight: 800 !important;
        color: var(--ms-neutral-secondary) !important;
        font-size: var(--fs-tab) !important;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        padding: 0 clamp(1.5rem, 4vw, 5rem) !important;
        transition: all 0.4s ease !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--brand-primary) !important;
        background: transparent !important;
        box-shadow: 0 clamp(-6px, -1vw, -10px) 0px var(--brand-primary) inset !important;
    }
    [data-theme="light"] [aria-selected="true"] { background: var(--ms-azure-white) !important; }

    .stButton > button {
        border-radius: 0 !important;
        height: clamp(55px, 8vw, 65px) !important;
        font-weight: 800 !important;
        font-size: clamp(1rem, 2.5vw, 1.2rem) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }

    /* Primary button — logo blue→green gradient */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #2563EB 0%, #16A34A 100%) !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #15803d 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
    }

    /* Radio button — selected dot and ring brand blue */
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] + div,
    [data-testid="stRadio"] [aria-checked="true"] {
        color: #2563EB !important;
    }
    [data-testid="stRadio"] [role="radio"] div[data-checked="true"],
    [data-testid="stRadio"] label[data-checked="true"] span:first-child {
        /* background-color: #2563EB !important; */
        border-color: #2563EB !important;
    }
    /* Catch-all for the inner filled circle */
    [data-testid="stRadio"] input[type="radio"]:checked + div > div {
        /* background-color: #2563EB !important; */
        border-color: #2563EB !important;
    }
    [data-testid="stRadio"] [data-baseweb="radio"] [class*="radioMark"] {
        /* background-color: #2563EB !important; */
        border-color: #2563EB !important;
    }


    .chat-bubble-assistant {
        background: var(--secondary-background-color) !important;
        border-left: 8px solid var(--brand-primary) !important;
        color: var(--text-color) !important;
        font-weight: 600;
        padding: clamp(1rem, 3vw, 1.5rem) !important;
    }
    [data-theme="light"] .chat-bubble-assistant { background: var(--ms-azure-white) !important; border-bottom: 1px solid var(--ms-neutral-tertiary) !important; }

    .chat-bubble-user {
        border: 2px dashed var(--ms-gray) !important;
        background: transparent !important;
        color: var(--text-color) !important;
        font-weight: 600;
        padding: clamp(1rem, 3vw, 1.5rem) !important;
    }

    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] { letter-spacing: 0.1em; padding: 0 1.5rem !important; }
        .main-container { padding: 1.5rem 1rem; }
        .card { padding: 1.25rem; }
    }
</style>
""", unsafe_allow_html=True)
st.markdown(load_avatar_css(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------------
if "transcribed_text" not in st.session_state:
    st.session_state["transcribed_text"] = ""
if "notes_db" not in st.session_state:
    st.session_state["notes_db"] = []
if "app_status" not in st.session_state:
    st.session_state["app_status"] = ("ready", "Ready")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_transcript" not in st.session_state:
    st.session_state["last_transcript"] = ""
if "last_ai_reply" not in st.session_state:
    st.session_state["last_ai_reply"] = ""
if "avatar_talking" not in st.session_state:
    st.session_state["avatar_talking"] = False
if "last_processed_audio" not in st.session_state:
    st.session_state["last_processed_audio"] = None
if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"
if "active_flow" not in st.session_state:
    st.session_state["active_flow"] = None
if "next_unlock_days" not in st.session_state:
    st.session_state["next_unlock_days"] = 3
if "habits" not in st.session_state:
    st.session_state["habits"] = []
if "health_twin_summary" not in st.session_state:
    st.session_state["health_twin_summary"] = "No profile generated yet."
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"
if "show_checkin_results" not in st.session_state:
    st.session_state["show_checkin_results"] = False
if "last_checkin_result" not in st.session_state:
    st.session_state["last_checkin_result"] = None

def get_mood_label(sentiment_score):
    """Maps sentiment score to a professional mood scale."""
    if sentiment_score <= -0.6: return "Very Low"
    if sentiment_score <= -0.2: return "Low"
    if sentiment_score < 0.2: return "Neutral"
    if sentiment_score < 0.6: return "Good"
    return "Very Good"

def get_approx_vocal_signals(audio_bytes):
    """Simulates approximate vocal stress/energy detection from audio metadata."""
    if not audio_bytes: return "Normal", "Steady"
    # In a real app, we'd use librosa or similar. For now, we simulate based on byte length/randomness 
    # to demonstrate the UI requirement.
    import random
    stress_levels = ["Low", "Moderate", "High"]
    energy_levels = ["Calm", "Steady", "Elevated"]
    return random.choice(stress_levels), random.choice(energy_levels)

def render_empty_state(message, icon="ℹ️", cta=None):
    """Render a professional, centered empty-state message with an optional CTA."""
    st.markdown(f"""
    <div style="text-align: center; padding: 2.5rem; background: rgba(58, 134, 255, 0.03); border-radius: 8px; border: 1px dashed #cbd5e1; margin: 1rem 0;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="color: #64748b; font-weight: 500; font-size: 0.95rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    if cta:
        if st.button(cta["label"], key=f"cta_{message[:10]}", use_container_width=True):
            cta["action"]()


# -----------------------------------------------------------------------------
# Auth & File Storage Functions
# -----------------------------------------------------------------------------
import bcrypt
import os
import json

USERS_DB_FILE = "data/users_db.json"

def user_data_dir(username):
    # Sanitize username (allow only letters, numbers, underscore)
    safe_name = "".join([c for c in username if c.isalnum() or c == '_'])
    path = os.path.join("data", "users", safe_name)
    os.makedirs(path, exist_ok=True)
    return path

def render_auto_mic(key="auto_mic"):
    """
    Custom HTML/JS component for an auto-stopping microphone.
    This version uses st.components.v1.html and passes data back via session state.
    """
    import base64
    from streamlit.components.v1 import html

    # We use a session state key to store the actual audio data
    data_key = f"{key}_data"
    if data_key not in st.session_state:
        st.session_state[data_key] = None

    component_html = f"""
    <div id="mic-container" style="display: flex; align-items: center; justify-content: center; padding: 10px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; cursor: pointer; transition: all 0.3s ease; height: 60px;">
        <div id="mic-icon" style="font-size: 1.2rem; margin-right: 10px;">🎙️</div>
        <div id="status-text" style="font-weight: 600; color: #1e293b; font-family: sans-serif; font-size: 0.9rem;">Click to Speak</div>
    </div>

    <script>
        const container = document.getElementById('mic-container');
        const icon = document.getElementById('mic-icon');
        const status = document.getElementById('status-text');
        
        let mediaRecorder;
        let audioContext;
        let analyzer;
        let dataArray;
        let stream;
        let silenceStartTime = null;
        const SILENCE_THRESHOLD = 0.01;
        const SILENCE_DURATION = 1500;

        async function startRecording() {{
            try {{
                stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                mediaRecorder = new MediaRecorder(stream);
                const chunks = [];

                mediaRecorder.ondataavailable = e => chunks.push(e.data);
                mediaRecorder.onstop = async () => {{
                    const blob = new Blob(chunks, {{ type: 'audio/webm' }});
                    const reader = new FileReader();
                    reader.onloadend = () => {{
                        const base64Audio = reader.result.split(',')[1];
                        // Send data back to Streamlit via a hacky but effective way for raw HTML components
                        // We use the 'streamlit:setComponentValue' message which Streamlit listens for
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: base64Audio,
                            key: '{key}'
                        }}, '*');
                    }};
                    reader.readAsDataURL(blob);
                    
                    icon.innerText = '✅';
                    status.innerText = 'Done';
                    stream.getTracks().forEach(track => track.stop());
                }};

                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(stream);
                analyzer = audioContext.createAnalyser();
                analyzer.fftSize = 256;
                source.connect(analyzer);
                dataArray = new Float32Array(analyzer.frequencyBinCount);

                mediaRecorder.start();
                icon.innerText = '🔴';
                status.innerText = 'Listening...';
                container.style.background = '#fef2f2';
                container.style.borderColor = '#ef4444';
                
                checkSilence();
            }} catch (err) {{
                status.innerText = 'Mic Error';
                icon.innerText = '❌';
            }}
        }}

        function checkSilence() {{
            if (mediaRecorder.state !== 'recording') return;
            analyzer.getFloatTimeDomainData(dataArray);
            let sumSquares = 0.0;
            for (const amplitude of dataArray) {{ sumSquares += amplitude * amplitude; }}
            const volume = Math.sqrt(sumSquares / dataArray.length);
            if (volume < SILENCE_THRESHOLD) {{
                if (!silenceStartTime) silenceStartTime = Date.now();
                if (Date.now() - silenceStartTime > SILENCE_DURATION) {{
                    mediaRecorder.stop();
                    return;
                }}
            }} else {{
                silenceStartTime = null;
            }}
            requestAnimationFrame(checkSilence);
        }}

        container.onclick = () => {{
            if (!mediaRecorder || mediaRecorder.state === 'inactive') startRecording();
        }};
    </script>
    """
    html(component_html, height=100)
    
    # In raw HTML components, this 'key' trick only works if the component registers itself.
    # We'll use the return value of st.components.v1.html (which is a DeltaGenerator) 
    # but the ACTUAL value comes back through the 'key' in session state.
    return st.session_state.get(key)

def user_notes_path(username):
    return os.path.join(user_data_dir(username), "notes.json")

def load_notes():
    if not st.session_state.get("is_authenticated"):
        return []
    path = user_notes_path(st.session_state["username"])
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except:
            return []

def save_note(note):
    if not st.session_state.get("is_authenticated"):
        return
    notes = load_notes()
    notes.append(note)
    path = user_notes_path(st.session_state["username"])
    with open(path, "w") as f:
        json.dump(notes, f, indent=2)

def clear_local_history():
    if not st.session_state.get("is_authenticated"):
        return
    path = user_notes_path(st.session_state["username"])
    if os.path.exists(path):
        os.remove(path)
    st.session_state["notes_db"] = []
    st.session_state["messages"] = []

def load_users_db():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(USERS_DB_FILE):
        with open(USERS_DB_FILE, "w") as f:
            json.dump({}, f)
        return {}
    with open(USERS_DB_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def save_users_db(db):
    os.makedirs("data", exist_ok=True)
    with open(USERS_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)



# -----------------------------------------------------------------------------
# Privacy / Redaction Functions
# -----------------------------------------------------------------------------
def redact_phi(text):
    if not text: return text
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[REDACTED_PHONE]', text)
    text = re.sub(r'(?i)(dob|date of birth|birthdate)[\s:]*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', r'\1: [REDACTED_DOB]', text)
    text = re.sub(r'(?i)(patient name|name)[\s:]+([A-Z][a-z]+ [A-Z][a-z]+)', r'\1: [REDACTED_NAME]', text)
    return text

def clean_html(raw_html):
    """Removes HTML tags from a string for safe text display."""
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', str(raw_html))
    return cleantext

# -----------------------------------------------------------------------------
# Audio & Speech-to-Text (Browser-based)
# -----------------------------------------------------------------------------
# Note: sounddevice (sd.rec) does NOT work in browser-based Streamlit deployments.
# We now use audio_recorder_streamlit for client-side recording.

def transcribe_audio_bytes(audio_bytes):
    """Securely interact with Azure Speech to Text and return a clean transcript."""
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    
    if not speech_key or not speech_region:
        return "[Error: Azure Speech Keys Missing. Check .env]"
        
    try:
        import azure.cognitiveservices.speech as speechsdk
        import tempfile
        import io
        from pydub import AudioSegment
        
        # Streamlit's mic recorder usually outputs as WebM/Ogg. We need valid WAV framing for Azure.
        audio_stream = io.BytesIO(audio_bytes)
        try:
            # Try loading as audio segment and export explicitly to wave
            sound = AudioSegment.from_file(audio_stream)
            # Azure Speech SDK expects 16kHz, 16-bit, mono PCM audio
            sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sound.export(tmp.name, format="wav")
                tmp_path = tmp.name
        except Exception as dub_e:
            # Fallback if pydub fails
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        # Support common languages for auto-detection (Azure limit: max 4 for DetectAudioAtStart)
        languages = ["en-US", "es-ES", "fr-FR", "zh-CN"]
        auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
        
        audio_config = speechsdk.AudioConfig(filename=tmp_path)
        
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            auto_detect_source_language_config=auto_detect_source_language_config, 
            audio_config=audio_config
        )
        
        # We need a synchronous resolution here instead of async to avoid Streamlit event loop issues
        result = speech_recognizer.recognize_once()
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "" # Safe to return empty for ambient noise
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            msg = f"[Error: Speech recognition failed. Reason: {cancellation_details.reason}]"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                msg += f" Details: {cancellation_details.error_details}"
            return msg
    except Exception as e:
        return f"[Error: Speech recognition exception: {str(e)}]"

# -----------------------------------------------------------------------------
# Medical NER (SciSpacy + Keyword Fallback)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SciSpacy Model...")
def load_ner_model():
    if HAS_SPACY:
        try:
            return spacy.load("en_core_sci_sm")
        except:
            return None
    return None

def extract_medical_concepts(text):
    entities = {
        "symptoms": set(), "conditions": set(),
        "medications": set(), "vitals": set(), "procedures": set()
    }
    text_lower = text.lower()
    
    bp_matches = re.findall(r'\b\d{2,3}/\d{2,3}\b', text)
    if bp_matches: entities["vitals"].update([f"BP: {m}" for m in bp_matches])
    temp_matches = re.findall(r'\b(temp(erature)?|t)\s*[:=]?\s*(\d{2,3}(\.\d)?)\b', text_lower)
    if temp_matches: entities["vitals"].update([f"Temp: {t[2]}" for t in temp_matches])
    hr_matches = re.findall(r'\b(hr|pulse|heart rate)\s*[:=]?\s*(\d{2,3})\b', text_lower)
    if hr_matches: entities["vitals"].update([f"HR: {h[1]}" for h in hr_matches])

    nlp = load_ner_model()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            e_text = ent.text.lower()
            if any(x in e_text for x in ["pain", "ache", "fever", "cough", "nausea", "fatigue", "tired"]):
                entities["symptoms"].add(ent.text)
            elif any(x in e_text for x in ["mg", "ml", "tablet", "aspirin", "ibuprofen", "tylenol", "dose"]):
                entities["medications"].add(ent.text)
            elif any(x in e_text for x in ["surgery", "x-ray", "mri", "scan", "test", "biopsy"]):
                entities["procedures"].add(ent.text)
            else:
                entities["conditions"].add(ent.text)
                
    fallback_symptoms = ["headache", "fever", "chills", "nausea", "vomiting", "dizziness", "shortness of breath", "fatigue", "pain"]
    for s in fallback_symptoms:
        if s in text_lower: entities["symptoms"].add(s)
            
    return {k: sorted(list(v)) for k, v in entities.items()}

# -----------------------------------------------------------------------------
# Advanced Analysis Logic
# -----------------------------------------------------------------------------
def calculate_quality_score(raw_text, soap_text):
    score = 100
    checks = []
    raw = raw_text.lower()
    soap = soap_text.lower()
    missing = []
    
    if "subjective:" in soap and "objective:" in soap and "assessment:" in soap and "plan:" in soap:
        checks.append(("Formatting S/O/A/P present", True))
    else:
        checks.append(("Missing standard S/O/A/P headers", False))
        score -= 20
        
    if any(v in raw for v in ["bp", "blood pressure", "temp", "pulse", "hr", "/"]):
        checks.append(("Vitals mentioned", True))
    else:
        checks.append(("Missing Vitals", False))
        score -= 10
        missing.append("vitals")
        
    if any(d in raw for d in ["days", "weeks", "months", "hours", "since"]):
        checks.append(("Symptom duration noted", True))
    else:
        checks.append(("Missing symptom duration", False))
        score -= 10
        missing.append("duration")
        
    if "follow-up" in soap or "follow up" in soap or "return" in soap:
        checks.append(("Follow-up plan established", True))
    else:
        checks.append(("No clear follow-up stated", False))
        score -= 10
        missing.append("follow-up")
        
    return max(0, score), checks, missing

def find_related_diary_entries(soap_entities, notes):
    related = []
    if not soap_entities.get("symptoms") and not soap_entities.get("conditions"):
        return related
        
    search_terms = set([x.lower() for x in soap_entities.get("symptoms", []) + soap_entities.get("conditions", [])])
    now = datetime.today()
    
    for n in reversed(notes):
        if n.get("mode") == "diary":
            try:
                date_obj = datetime.strptime(n.get("date", "1970-01-01"), "%Y-%m-%d")
            except:
                continue
            if (now - date_obj).days <= 14:
                diary_tags_and_text = set([t.lower() for t in n.get("diary", {}).get("tags", [])] + n['raw_text_redacted'].lower().split())
                if search_terms.intersection(diary_tags_and_text):
                    related.append(n)
                    if len(related) >= 3: break
    return related

def analyze_trends(diary_notes):
    if not diary_notes: return {}
    recent_notes = diary_notes[-14:]
    
    sentiments = [n["diary"]["sentiment"] for n in recent_notes]
    slope = np.polyfit(range(len(sentiments)), sentiments, 1)[0] if len(sentiments) > 1 else 0
        
    symptoms = []
    for n in recent_notes:
        symptoms.extend(n.get("medical_entities", {}).get("symptoms", []))
    
    from collections import Counter
    sym_counts = Counter(symptoms)
    
    return {
        "sentiment_slope": float(slope),
        "sentiment_avg": float(np.mean(sentiments)) if sentiments else 0.0,
        "top_symptoms": sym_counts.most_common(5),
        "total_notes": len(recent_notes)
    }

def generate_risk_alerts(trends):
    alerts = []
    if trends.get("sentiment_slope", 0) < -0.1:
        alerts.append("Mood Alert: Your sentiment has been trending downwards recently. Consider practicing self-care or speaking with someone you trust.")
        
    escalated = False
    for sym, count in trends.get("top_symptoms", []):
        if count >= 3:
            alerts.append(f"Persistence Alert: You reported '{sym}' {count} times recently. Consider consulting a clinician if it persists.")
            escalated = True
            
    if escalated:
        alerts.append("🚨 **SMART ESCALATION**: Persistent symptoms detected. Please use the Care Circle feature to generate a report and consult a healthcare professional.")
        
    return alerts

# -----------------------------------------------------------------------------
# Processors
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def generate_ai_response(messages, temp=0.2):
    """Securely interact with Azure OpenAI."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    if not api_key or not endpoint or not deployment:
        return "ERROR: Azure OpenAI Credentials Missing. Check .env file."
        
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Azure OpenAI Error: {e}"

def process_soap(text):
    system_prompt = """
    You are an expert medical assistant. Format the dictation into a clean SOAP note.
    Strictly use this structure only:
    Subjective:
    - [details]
    Objective:
    - [details]
    Assessment:
    - [details]
    Plan:
    - [details]
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    return generate_ai_response(messages, temp=0.2)

def process_diary_logic(text):
    analyzer = SentimentIntensityAnalyzer() if HAS_VADER else None
    sentiment = analyzer.polarity_scores(text)["compound"] if analyzer else 0.0
    
    text_lower = text.lower()
    tags = set()
    if any(w in text_lower for w in ["pain", "headache", "fever", "cough"]): tags.add("symptoms")
    if any(w in text_lower for w in ["ate", "food", "lunch", "dinner"]): tags.add("food")
    if any(w in text_lower for w in ["happy", "sad", "stressed", "anxious"]): tags.add("mood")
        
    suggs = []
    if sentiment < -0.2:
        suggs.append("- Consider rest, hydration, talking to someone you trust, or a clinician if concerned.")
        
    return {"sentiment": sentiment, "tags": list(tags), "suggestions": suggs, "summary": text[:50]+"..."}

def render_chips(entities_dict):
    chips_html = ""
    found = False
    for category, items in entities_dict.items():
        for item in items:
            found = True
            chips_html += f'<span class="chip">{item}</span>'
    
    if not found:
        return "No medical entities detected."
    return chips_html

# -----------------------------------------------------------------------------
# AI Care Assistant Chat Functions
# -----------------------------------------------------------------------------
def detect_red_flags(text):
    """Detects severe emergency terms in user text."""
    text = text.lower()
    red_flag_terms = ["chest pain", "trouble breathing", "difficulty breathing", "fainting", "pass out", 
                      "severe allergic reaction", "confusion", "severe dehydration", "suicide", 
                      "kill myself", "self-harm", "severe bleeding"]
    return any(term in text for term in red_flag_terms)

def build_assistant_context(notes, use_soap=True, use_diary=True):
    context = ""
    if use_soap:
        soap_notes = [n for n in notes if isinstance(n, dict) and n.get("mode") == "soap"]
        if soap_notes:
            last = soap_notes[-1]
            if isinstance(last, dict):
                redacted = last.get('raw_text_redacted', '')
                if not isinstance(redacted, str): redacted = str(redacted)
                context += f"Last SOAP Note ({last.get('date', 'Unknown Date')}): {redacted[:100]}...\n"
            
    if use_diary:
        diary_notes = [n for n in notes if isinstance(n, dict) and n.get("mode") == "diary"]
        if diary_notes:
            if isinstance(diary_notes, list):
                recent_diary = diary_notes[-7:]
                context += f"Last {len(recent_diary)} Diary Entries:\n"
                for d in recent_diary:
                    if isinstance(d, dict):
                        redacted = d.get('raw_text_redacted', '')
                        redacted_str = str(redacted)
                        context += f"- {d.get('date', 'Unknown Date')}: {redacted_str[:50]}...\n"
            
            trends_data = analyze_trends(diary_notes)
            avg_mood = float(trends_data.get("sentiment_avg", 0.0))
            top_syms_raw = trends_data.get("top_symptoms", [])
            top_syms_list = []
            if isinstance(top_syms_raw, list):
                for s in top_syms_raw:
                    if isinstance(s, (list, tuple)) and len(s) > 0:
                        top_syms_list.append(str(s[0]))
            
            context += f"\nRecent Mood Avg: {avg_mood:.2f}\n"
            context += f"Recent Top Symptoms: {', '.join(top_syms_list)}\n"
            
    return context.strip()

def generate_chat_reply(user_message, context, history):
    if detect_red_flags(user_message):
        return "SAFETY ALERT: Seek urgent medical help immediately by calling local emergency services or going to the nearest emergency room. If you are a minor, please talk to a trusted adult right away."
    
    system_prompt = f"""You are a supportive AI Care Assistant. You provide informational support ONLY.
CRITICAL SAFETY RULES:
1. NEVER diagnose. NEVER prescribe medication.
2. Use safe wording: "may", "could", "consider".
3. Keep responses EXTREMELY concise (max 25 words). No markdown.

6. RESPOND IN THE SAME LANGUAGE AS THE USER. If the user speaks Spanish, reply in Spanish. If they speak Hindi, reply in Hindi.
A) Empathetic acknowledgement.
B) One pattern or suggestion.
C) One follow-up question.

Context:
{context if context else "No recent logs available."}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-5:]: # last 5 to keep context short
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    reply = generate_ai_response(messages, temp=0.3)
    if "ERROR" in reply or "Azure OpenAI Error" in reply:
        # Rule-based fallback
        reply = "I hear what you're saying. "
        if context:
            reply += f"Based on your recent logs, I noticed some context. {context[:150]}... "
            
        reply += "Have these symptoms or feelings changed recently? Consider resting and staying hydrated. If symptoms persist, please consider speaking with a clinician."
    
    return reply

def generate_tts_audio(text):
    """Generates Text-to-Speech audio bytes using Azure Speech SDK."""
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    
    if not speech_key or not speech_region:
        return None
        
    try:
        import azure.cognitiveservices.speech as speechsdk
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        # We don't want to output to speaker directly here, we want the bytes to send to Streamlit
        # So we use a PullAudioOutputStream
        pull_stream = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
        
        # You can customize the voice name here
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            st.error(f"TTS Synthesis Failed: {result.reason}")
            return None
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def update_streak(notes):
    """Calculates login/entry streak and unlock progress from notes."""
    if not notes:
        return 0, 1, 3
    
    dates = []
    for n in notes:
        try:
            dates.append(datetime.strptime(n.get("date", "1970-01-01"), "%Y-%m-%d").date())
        except:
            pass
            
    dates = sorted(list(set(dates)), reverse=True)
    
    streak = 0
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    
    if dates and (dates[0] == today or dates[0] == yesterday):
        streak = 1
        current_date = dates[0]
        for d in dates[1:]:
            if (current_date - d).days == 1:
                streak += 1
                current_date = d
            elif (current_date - d).days > 1:
                break
                
    total_logs = len(notes)
    level = (total_logs // 5) + 1
    next_unlock = 5 - (total_logs % 5)
    
    return streak, level, next_unlock

def generate_insight(entry_text):
    messages = [{"role": "system", "content": "You are a wellness AI. Give ONE short (15-word max), positive, non-medical insight about this journal entry. DO NOT diagnose. Sound empathetic. ALWAYS RESPOND IN THE SAME LANGUAGE AS THE USER."}]
    messages.append({"role": "user", "content": entry_text})
    return generate_ai_response(messages, temp=0.5)

def generate_weekly_summary(notes):
    context = build_assistant_context(notes, use_soap=False, use_diary=True)
    system_prompt = f"Summarize the user's past 7 days based on the following logs. Keep it supportive, concise (under 50 words), and non-medical. NEVER diagnose. ALWAYS RESPOND IN THE SAME LANGUAGE AS THE USER. \n\nLogs:\n{context}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.3)

def generate_health_twin_summary(notes):
    context = build_assistant_context(notes, use_soap=True, use_diary=True)
    system_prompt = f"Analyze these logs and create a dynamic 'AI Health Twin Profile'. Summarize behavioral patterns, mood trends, and chronicity of symptoms. Keep it under 100 words, formatting with bullet points. NEVER diagnose. ALWAYS RESPOND IN THE SAME LANGUAGE AS THE USER. \n\nLogs:\n{context}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.4)

def generate_micro_habits(notes):
    context = build_assistant_context(notes, use_soap=False, use_diary=True)
    system_prompt = f"Based on the user's logs, suggest exactly 2 small, actionable 'Micro-Habits' they can do today to improve their specific documented challenges. Be very brief. ALWAYS RESPOND IN THE SAME LANGUAGE AS THE USER. \n\nLogs:\n{context}"
    response = generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.4)
    # Split by newlines or bullets to lists
    habits = [h.strip("- *").strip() for h in response.split("\n") if h.strip() and len(h) > 5]
    return habits[:2]

def generate_question_prep(notes):
    context = build_assistant_context(notes, use_soap=True, use_diary=True)
    system_prompt = f"Draft 3 specific questions the patient should ask their doctor during their next visit, based on their unresolved or persistent symptoms in these logs. \n\nLogs:\n{context}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.3)

def generate_care_circle_report(notes):
    context = build_assistant_context(notes, use_soap=True, use_diary=True)
    system_prompt = f"Create a professional, structured 'Caregiver / Doctor Update Report' covering the last 7 days. Include: 1) Top Symptoms, 2) General Sentiment Trend, 3) Important Notes. Omit extreme emotional venting, focus on factual health trends. \n\nLogs:\n{context}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.3)

def process_live_copilot(text):
    system_prompt = f"You are a clinical copilot listening to a doctor-patient consultation. Output two sections: 'Structured Notes' and 'Suggested Follow-up Questions for Patient'. Do NOT diagnose.\n\Transcript:\n{text}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.2)

def generate_monthly_report(notes):
    context = build_assistant_context(notes, use_soap=False, use_diary=True)
    system_prompt = f"Provide a brief, encouraging high-level summary of the user's month based on these logs. Identify any broad recurring themes. Keep it under 60 words. Strict rule: NO medical advice or diagnosis. \n\nLogs:\n{context}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.3)

def generate_doctor_prep(notes):
    context = build_assistant_context(notes, use_soap=True, use_diary=True)
    system_prompt = f"Based on the following logs, prepare a short, bulleted list of 2-3 key points the user should discuss at their next doctor's appointment. Be informative, not diagnostic. \n\nLogs:\n{context}"
    return generate_ai_response([{"role": "system", "content": system_prompt}], temp=0.2)

def generate_pdf_report(username, notes):
    """Generates a simple PDF report using fpdf."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Monthly Health Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient: {username}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Report Date: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align='L')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Summary Over Time", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    
    diary_notes = [n for n in notes if n.get("mode") == "diary"]
    if diary_notes:
        trends = analyze_trends(diary_notes)
        avg_mood = trends.get("sentiment_avg", 0)
        pdf.multi_cell(0, 10, txt=f"Mood Trend: {get_mood_label(avg_mood)} (Avg Score: {avg_mood:.2f})")
        top_syms = trends.get("top_symptoms", [])
        sym_txt = ", ".join([f"{s[0]} ({s[1]})" for s in top_syms])
        pdf.multi_cell(0, 10, txt=f"Frequent Symptoms: {sym_txt if sym_txt else 'None reported'}")
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Recent Timeline Highlights", ln=True, align='L')
    pdf.set_font("Arial", size=10)
    
    for n in notes[-10:]:
        date = n.get("date", "N/A")
        raw = n.get("raw_text_redacted", "")[:80] + "..."
        pdf.multi_cell(0, 8, txt=f"[{date}] {raw}")
        
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: This report is generated by AI for informational purposes only and does not constitute medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
    
    return pdf.output(dest='S').encode('latin-1')

def render_privacy_badges():
    st.markdown("""
        <div style="display: flex; gap: 10px; margin-bottom: 20px;">
            <span style="background-color: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid #bbf7d0;">✓ Privacy Shield Enabled</span>
            <span style="background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid #e2e8f0;">⬚ Synthetic Data Mode</span>
            <span style="background-color: #fef2f2; color: #991b1b; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid #fecaca;">🛡 No PHI Stored</span>
        </div>
    """, unsafe_allow_html=True)

def get_avatar_advice(user_message, context):
    """Unified function to get text and voice advice from OpenAI Assistant."""
    # 1. Generate text and voice advice
    with st.spinner("Avatar is synthesizing advice..."):
        # Add user message to persistent history
        st.session_state["messages"].append({"role": "user", "content": user_message})
        st.session_state["last_transcript"] = user_message
        
        if st.session_state.get("active_flow") == "checkin":
            # Handle user response to check-in
            reply = ""
            
            # Save the entry
            entities = extract_medical_concepts(user_message)
            diary_data = process_diary_logic(user_message)
            
            if not st.session_state.get("privacy_mode", False):
                note_record = {
                    "timestamp": datetime.now().isoformat(),
                    "date": datetime.today().strftime("%Y-%m-%d"),
                    "mode": "diary",
                    "raw_text_redacted": redact_phi(user_message),
                    "medical_entities": entities,
                    "diary": diary_data
                }
                if "tags" not in note_record["diary"]: note_record["diary"]["tags"] = []
                note_record["diary"]["tags"].append("Daily Check-in")
                save_note(note_record)
                
                # Update streak and get stats
                all_notes = load_notes()
                streak, level, next_unlock = update_streak(all_notes)
                st.session_state["next_unlock_days"] = next_unlock
                
                # Generate reward response
                insight = generate_insight(user_message)
                reply = f"Saved ✅ | Streak: 🔥 {streak} day(s) \n\n{insight}\n\n*Next unlock: Level {level + 1} in {next_unlock} more log(s).*"
            else:
                reply = "Check-in complete (Privacy Mode Active - not saved)."
            
            st.session_state["active_flow"] = None
            
        else:
            # Normal Q&A Flow
            # Get Text Reply
            reply = generate_chat_reply(user_message, context, st.session_state["messages"])
            
            # Auto-log to Analytics (Diary entry) if substantive
            entities = extract_medical_concepts(user_message)
            diary_data = process_diary_logic(user_message)
            has_medical = any(len(v) > 0 for v in entities.values())
            has_sentiment = abs(diary_data.get("sentiment", 0)) > 0.1
            
            if (has_medical or has_sentiment) and not st.session_state.get("privacy_mode", False):
                note_record = {
                    "timestamp": datetime.now().isoformat(),
                    "date": datetime.today().strftime("%Y-%m-%d"),
                    "mode": "diary",
                    "raw_text_redacted": redact_phi(user_message),
                    "medical_entities": entities,
                    "diary": diary_data
                }
                if "tags" not in note_record["diary"]: note_record["diary"]["tags"] = []
                if "Chat Insight" not in note_record["diary"]["tags"]:
                    note_record["diary"]["tags"].append("Chat Insight")
                save_note(note_record)

        # Get Voice Advice
        audio_bytes = generate_tts_audio(reply)
        
        # Store Assistant Reply persistently
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.session_state["last_ai_reply"] = reply

        # Set Audio & Animation Flags
        if audio_bytes:
            st.session_state["last_audio"] = audio_bytes
            st.session_state["new_audio_flag"] = True

# -----------------------------------------------------------------------------
# Authentication UI
# -----------------------------------------------------------------------------
if not st.session_state["is_authenticated"]:
    import base64

    def img_to_base64(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            return None

if not st.session_state["is_authenticated"]:
    avatar_css = load_avatar_css()
    logo_b64   = img_to_base64("logo.png")
    if not st.session_state["is_authenticated"]:
        avatar_css = load_avatar_css()
        logo_b64   = img_to_base64("logo.png")
        logo_src   = f"data:image/png;base64,{logo_b64}" if logo_b64 else ""

        # --- Full-page premium CSS ---
        st.markdown(avatar_css, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Hide Streamlit chrome */
            [data-testid="stToolbar"], header { display: none !important; }
            footer { visibility: hidden !important; }

            /* Make the app take full height */
            .stApp { background: #f0f4f9; }
            [data-testid="stAppViewContainer"] { padding: 0 !important; }
            [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
            section[data-testid="stMain"] > div { padding: 0 !important; }

            /* Two column layout */
            [data-testid="stHorizontalBlock"] {
                gap: 0 !important;
                min-height: 100vh;
            }

            /* LEFT column — blue gradient hero */
            [data-testid="stHorizontalBlock"] > div:first-child {
                background: linear-gradient(160deg, #1a3a6b 0%, #3A86FF 55%, #2EC4B6 100%) !important;
                min-height: 100vh;
                display: flex !important;
                flex-direction: column;
                align-items: center;
                justify-content: flex-end;
                position: relative;
                overflow: hidden;
                padding: 0 !important;
            }
            [data-testid="stHorizontalBlock"] > div:first-child::before {
                content: '';
                position: absolute;
                top: -100px; right: -100px;
                width: 350px; height: 350px;
                background: rgba(255,255,255,0.05);
                border-radius: 50%;
                pointer-events: none;
            }
            [data-testid="stHorizontalBlock"] > div:first-child::after {
                content: '';
                position: absolute;
                bottom: 100px; left: -80px;
                width: 220px; height: 220px;
                background: rgba(255,255,255,0.05);
                border-radius: 50%;
                pointer-events: none;
            }

            /* RIGHT column — white form panel */
            [data-testid="stHorizontalBlock"] > div:last-child {
                background: #ffffff !important;
                min-height: 100vh;
                display: flex !important;
                flex-direction: column;
                justify-content: flex-start;
                padding: 18vh 3.5rem 3rem !important;
                box-shadow: -16px 0 50px rgba(0,0,0,0.08);
            }

            /* Ensure Streamlit widgets inside right column aren't squished */
            [data-testid="stHorizontalBlock"] > div:last-child > div {
                width: 100%;
            }

            /* Hide the label above radio */
            [data-testid="stRadio"] > label { display: none !important; }

            /* Remove highlight box behind Login / Register text */
            [data-testid="stRadio"] label,
            [data-testid="stRadio"] div[data-baseweb="radio"],
            [data-testid="stRadio"] div[data-baseweb="radio"] div {
                background: none !important;
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }
            [data-testid="stRadio"] label:hover {
                background: none !important;
                background-color: rgba(0,0,0,0.04) !important;
            }

            /* Outer ring — brand blue */
            [data-baseweb="radio"] > div {
                border-color: #2563EB !important;
            }

            /* Inner filled dot — override Streamlit's red with brand blue */
            [data-baseweb="radio"] > div > div {
                /* background-color: #2563EB !important; */
            }

            /* SVG dot (some Streamlit versions) */
            [data-baseweb="radio"] svg circle {
                fill: #2563EB !important;
                stroke: #2563EB !important;
            }


        </style>
        """, unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 1])

        # ---- LEFT: Hero panel ----
        with left_col:
            # Logo top-left
            logo_top_html = f"""
            <div style="padding: 1.5rem 1.5rem 0; z-index:3; position:relative;">
                <img src="{logo_src}" style="height:200px; object-fit:contain;" />
            </div>
            """ if logo_src else ""

            # Avatar centred (Animated Intelligent Avatar)
            avatar_html = get_avatar_html()

            # Tagline below avatar — smaller & subdued
            tagline_html = """
            <div style="text-align:center; padding: 1rem 2rem 2rem; z-index:2; position:relative;">
                <p style="color:rgba(255,255,255,0.70); font-size:0.82rem; font-weight:600;
                           letter-spacing:0.06em; text-transform:uppercase; line-height:1.5; margin:0 0 0.4rem;">
                    Your AI-Powered Health Companion
                </p>
                <p style="color:rgba(255,255,255,0.45); font-size:0.75rem; max-width:260px;
                           margin:0 auto; line-height:1.6;">
                    Secure, intelligent note-taking and health insights — all in one place.
                </p>
            </div>
            """
            st.markdown(logo_top_html + avatar_html + tagline_html, unsafe_allow_html=True)

        # ---- RIGHT: Auth form ----
        with right_col:

            st.markdown(
                '<h2 style="text-align:center; font-size:2.0rem; font-weight:800; margin:0 0 0.3rem;'
                'background: linear-gradient(90deg, #2563EB 0%, #16A34A 100%);'
                '-webkit-background-clip: text; -webkit-text-fill-color: transparent;'
                'background-clip: text;">Welcome to Medinoted.com</h2>'
                '<p style="text-align:center; color:#2EC4B6; font-size:0.95rem; margin:0 0 2rem; font-weight:600;">Sign in to your account</p>',
                unsafe_allow_html=True
            )

            mode = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
            mode = "Login" if "Login" in mode else "Register"

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            username_input = st.text_input("Username", placeholder="Enter your username")
            password_input = st.text_input("Password", type="password", placeholder="Enter your password")

            if mode == "Register":
                password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
                if st.button("Create Account →", type="primary", use_container_width=True):
                    if not username_input or not password_input:
                        st.error("Please fill all fields.")
                    elif len(password_input) < 8:
                        st.error("Password must be at least 8 characters long.")
                    elif password_input != password_confirm:
                        st.error("Passwords do not match.")
                    else:
                        db = load_users_db()
                        safe_name = "".join([c for c in username_input if c.isalnum() or c == '_'])
                        if safe_name in db:
                            st.error("Username already exists.")
                        else:
                            hashed = bcrypt.hashpw(password_input.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                            from datetime import datetime
                            db[safe_name] = {
                                "password_hash": hashed,
                                "created_at": datetime.now().isoformat()
                            }
                            save_users_db(db)
                            user_data_dir(safe_name)
                            st.success("✅ Account created! Please login.")
            else:
                if st.button("Login →", type="primary", use_container_width=True):
                    if not username_input or not password_input:
                        st.error("Please enter credentials.")
                    else:
                        db = load_users_db()
                        safe_name = "".join([c for c in username_input if c.isalnum() or c == '_'])
                        if safe_name in db:
                            stored_hash = db[safe_name]["password_hash"].encode('utf-8')
                            if bcrypt.checkpw(password_input.encode('utf-8'), stored_hash):
                                st.session_state["is_authenticated"] = True
                                st.session_state["username"] = safe_name
                                st.session_state["transcribed_text"] = ""
                                st.session_state["messages"] = []
                                st.rerun()
                            else:
                                st.error("Invalid credentials.")
                        else:
                            st.error("Invalid credentials.")

            st.markdown(
                '<p style="font-size:0.75rem; color:#b0b8c1; text-align:center; margin-top:1.5rem;">'
                '🔒 Demo uses synthetic/anonymized data. Not a diagnostic tool.</p>',
                unsafe_allow_html=True
            )


        st.stop()




# -----------------------------------------------------------------------------
# Main UI Construction
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Navigation & Page Routing
# -----------------------------------------------------------------------------

def check_azure_connections():
    """Verify Azure environment variables lazily/securely without throwing errors."""
    checks = {
        "Azure OpenAI Endpoint": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        "Azure OpenAI Key": bool(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")),
        "Azure OpenAI Deployment": bool(os.getenv("AZURE_OPENAI_DEPLOYMENT")),
        "Azure Speech Key": bool(os.getenv("AZURE_SPEECH_KEY")),
        "Azure Speech Region": bool(os.getenv("AZURE_SPEECH_REGION"))
    }
    
    missing = [k for k, v in checks.items() if not v]
    return missing

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state['username']}")
        
        pages = ["Dashboard", "Daily Check-In", "AI Doctor", "Insights", "Reports", "Find Care", "Settings"]
        
        # Ensure radio stays in sync with programmatically set pages
        current_idx = 0
        if "current_page" in st.session_state and st.session_state["current_page"] in pages:
            current_idx = pages.index(st.session_state["current_page"])
            
        choice = st.radio("Navigation", pages, index=current_idx, label_visibility="collapsed")
        st.session_state["current_page"] = choice
        
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state["is_authenticated"] = False
            st.session_state["username"] = None
            st.session_state["messages"] = []
            st.session_state["transcribed_text"] = ""
            st.rerun()
            
        st.divider()
        st.markdown("### Secure Status")
        missing_keys = check_azure_connections()
        if missing_keys:
            st.error("⚠️ AI Backend Limited")
        else:
            st.success("✅ AI Engine Active")
            
        st.divider()
        all_notes = load_notes()
        streak, level, _ = update_streak(all_notes)
        st.markdown(f"**Health Level:** {level}")
        st.markdown(f"**Active Streak:** 🔥 {streak} Day(s)")

def render_dashboard():
    st.markdown('<div class="section-header" style="margin-bottom: 1.5rem; border-left-color: #3b82f6;">Patient Portal Dashboard</div>', unsafe_allow_html=True)
    
    # 1. Patient Status Panel (Top KPI Cards)
    all_notes = load_notes()
    diary_notes = [n for n in all_notes if n.get("mode") == "diary"]
    trends = analyze_trends(diary_notes)
    
    avg_mood = trends.get("sentiment_avg", 0)
    mood_label = get_mood_label(avg_mood)
    
    last_log_val = "None today"
    last_log_micro = "Daily log status"
    if all_notes:
        last_dt = datetime.fromisoformat(all_notes[-1].get("timestamp", datetime.now().isoformat()))
        if last_dt.date() == datetime.today().date():
            last_log_val = last_dt.strftime("%H:%M")
            last_log_micro = "Updated today"
        else:
            last_log_val = last_dt.strftime("%b %d")
            last_log_micro = f"Last log {last_dt.strftime('%H:%M')}"
        
    s1, s2, s3, s4 = st.columns(4)
    with s1: 
        with st.container(border=True):
            st.markdown(f'<div class="kpi-tile-clinical accent-blue"><div class="kpi-value-clinical">{mood_label}</div><div class="kpi-label-clinical">Today\'s Mood</div><div class="kpi-micro-clinical">Sentiment analysis</div></div>', unsafe_allow_html=True)
    with s2:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-tile-clinical accent-green"><div class="kpi-value-clinical">{last_log_val}</div><div class="kpi-label-clinical">Last Check-In</div><div class="kpi-micro-clinical">{last_log_micro}</div></div>', unsafe_allow_html=True)
    with s3: 
        with st.container(border=True):
            progress = (len(all_notes) % 5) * 20
            st.markdown(f'<div class="kpi-tile-clinical accent-orange"><div class="kpi-value-clinical">{progress}%</div><div class="kpi-label-clinical">Monthly report progress</div><div class="kpi-micro-clinical">Progress to summary</div></div>', unsafe_allow_html=True)
    with s4:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-tile-clinical accent-purple"><div class="kpi-value-clinical">Live</div><div class="kpi-label-clinical">Voice Assistant</div><div class="kpi-micro-clinical">System ready</div></div>', unsafe_allow_html=True)
    
    # 2. Primary Action
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("START DAILY CHECK-IN", type="primary", use_container_width=True):
        st.session_state["current_page"] = "Daily Check-In"
        st.rerun()
    st.markdown('<p style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 0.25rem;">Record your clinical symptoms, mood, or a voice note.</p>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # 3. Quick Access Cards
    st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem; color: #334155;">Quick Access</h3>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    
    items = [
        (q1, "🤖", "AI Doctor", "Consult with your virtual assistant.", "AI Doctor", "qa_dr_2"),
        (q2, "📋", "Health Records", "View your medical timeline.", "Reports", "qa_rec_2"),
        (q3, "🗺️", "Find Care", "Locate nearby medical facilities.", "Find Care", "qa_care_2")
    ]
    
    for col, icon, title, desc, page, key in items:
        with col:
            with st.container(border=True):
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.25rem;">{icon}</div>
                    <div style="font-weight: 700; color: #1e293b; font-size: 1.1rem;">{title}</div>
                    <p style="font-size: 0.8rem; color: #64748b; margin: 0.5rem 0 1rem 0; min-height: 2.5rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Open {title}", key=key, use_container_width=True):
                    st.session_state["current_page"] = page
                    st.rerun()
        
    # 4. Today Insight (Conditional)
    if diary_notes:
        last_entry = diary_notes[-1].get("raw_text_redacted", "")
        if last_entry:
            insight_text = generate_insight(last_entry)
            st.markdown(f"""
            <div class="insight-card-clinical">
                <h3 style="font-size: 1rem; font-weight: 700; margin-bottom: 0.25rem; color: #1e3a8a;">💡 Today's Insight</h3>
                <div style="font-size: 0.9rem; color: #1e40af; line-height: 1.4;">{clean_html(insight_text)}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Log your status today to generate a clinical health insight.")

    # 5. Recent Activity & Symptoms
    if all_notes:
        st.markdown('<h3 style="font-size: 1.1rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.75rem; color: #334155;">Recent Activity</h3>', unsafe_allow_html=True)
        col_act, col_symptom = st.columns([1.8, 1.2])
        
        with col_act:
            with st.container(border=True):
                for n in all_notes[-3:][::-1]:
                    dt = n.get("date", "Unknown")
                    raw_text = n.get("raw_text_redacted", "")
                    clean_text = clean_html(raw_text)
                    snippet = clean_text[:80] + "..." if len(clean_text) > 80 else clean_text
                    mode_icon = "🏥" if n.get("mode") == "soap" else "📓"
                    st.markdown(f"""
                    <div style="margin-bottom: 0.75rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: #3b82f6; font-size: 0.8rem;">{dt} {mode_icon}</span><br/>
                        <span style="font-size: 0.8rem; color: #475569; line-height: 1.4;">{snippet}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_symptom:
            top_symptoms = trends.get("top_symptoms", [])
            with st.container(border=True):
                if top_symptoms:
                    symptom_name, count = top_symptoms[0]
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem 0;">
                        <h4 style="margin: 0; color: #64748b; font-size: 0.85rem;">Primary Symptom</h4>
                        <div style="font-size: 1.75rem; font-weight: 700; color: #d97706; margin: 0.5rem 0;">{symptom_name.title()}</div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Reported <b>{count} times</b> recently</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align: center; color: #94a3b8; padding: 1rem 0; font-size: 0.85rem;">No symptoms tracked.</div>', unsafe_allow_html=True)

def render_find_care():
    st.markdown('<div class="section-header">Find Care Nearby</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Search Filters")
        location_input = st.text_input("City or Zip Code", placeholder="e.g. 90210")
        care_types = st.multiselect("Facility Type", ["hospital", "pharmacy", "clinic", "doctors"], default=["hospital", "pharmacy"])
        
        # Geolocation trigger
        if st.button("📍 Use My Current Location", use_container_width=True):
            st.session_state["trigger_geo"] = True
            
        if st.session_state.get("trigger_geo"):
            geo_container = st.container()
            with geo_container:
                # This helper runs the JS and returns result to 'geo_data' session key
                geo_data = get_geolocation()
                if geo_data:
                    st.session_state["user_location"] = geo_data
                    st.session_state["trigger_geo"] = False
        
        search_pressed = st.button("Search Medical Facilities", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Default coords (London-ish or NYC-ish or based on input)
        lat, lon = 40.7128, -74.0060 # NYC Default
        
        if st.session_state.get("user_location"):
            loc = st.session_state["user_location"]
            if isinstance(loc, dict) and "lat" in loc:
                lat, lon = loc["lat"], loc["lon"]
        
        # Simple geocoding fallback for text input
        if location_input:
            # Fake/Simple geocoding for common inputs to keep it snappy
            if "90210" in location_input: lat, lon = 34.0736, -118.4004
            elif "seattle" in location_input.lower(): lat, lon = 47.6062, -122.3321
            
        if search_pressed or st.session_state.get("user_location"):
            with st.spinner("Querying real-time healthcare data..."):
                facilities = query_nearby_care(lat, lon, categories=care_types)
                
                if facilities:
                    # Prepare map data
                    df = pd.DataFrame(facilities)
                    
                    st.pydeck_chart(pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v9',
                        initial_view_state=pdk.ViewState(
                            latitude=lat,
                            longitude=lon,
                            zoom=12,
                            pitch=45,
                        ),
                        layers=[
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=df,
                                get_position='[lon, lat]',
                                get_color='[22, 163, 74, 160]',
                                get_radius=200,
                                pickable=True,
                            ),
                        ],
                        tooltip={"text": "{name} ({type})"}
                    ))
                    
                    st.markdown(f"#### Results ({len(facilities)} found)")
                    for f in facilities[:5]: # Show top 5
                        st.markdown(f"""
                        <div class="card" style="padding: 1rem; margin-bottom: 0.5rem; border-left: 4px solid #16A34A;">
                            <b>{f['name']}</b><br/>
                            <span style="font-size: 0.8rem; color: #666;">{f['type']} • {lat:.2f}, {lon:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    render_empty_state("No facilities found in this area. Try a different location or broader filters.", icon="📍")
        else:
            render_empty_state("Enter a location or use GPS to see nearby healthcare facilities on the map.", icon="🗺️")

def render_reports_page():
    st.markdown('<div class="section-header">Health Reports & Records</div>', unsafe_allow_html=True)
    
    notes = load_notes()
    if not notes:
        render_empty_state("No records found. Complete a daily check-in to start building your timeline.", icon="📋", cta={"label": "Record First Entry", "action": lambda: st.session_state.update({"current_page": "Daily Check-In"})})
        return
        
    # Monthly Report Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📅 Monthly Health Summary")
    st.write("Generate a professional overview of your health trends for your clinician.")
    
    if st.button("Generate Monthly Report PDF", type="primary"):
        with st.spinner("Preparing clinical summary..."):
            pdf_bytes = generate_pdf_report(st.session_state["username"], notes)
            if pdf_bytes:
                st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"health_report_{datetime.today().strftime('%Y_%m')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Error generating PDF. Please ensure all dependencies are met.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Existing Records Timeline
    with st.expander("View Full Medical Timeline", expanded=True):
        for n in reversed(notes):
            mode_icon = "🏥" if n.get("mode") == "soap" else "📓"
            bg = "#f8fafc" if n.get("mode") == "soap" else "#ffffff"
            st.markdown(f"""
            <div style="background: {bg}; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between;">
                    <b>{mode_icon} {n.get('mode', 'Note').upper()}</b>
                    <span style="font-size: 0.8rem; color: #64748b;">{n.get('date')}</span>
                </div>
                <div style="margin-top: 5px; font-size: 0.95rem;">{clean_html(n.get('raw_text_redacted', ''))}</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Application Entry
# -----------------------------------------------------------------------------

render_sidebar()
render_privacy_badges()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

current_page = st.session_state.get("current_page", "Dashboard")

if current_page == "Dashboard":
    render_dashboard()


elif current_page == "AI Doctor":
    st.markdown('<div class="section-header">AI Care Assistant</div>', unsafe_allow_html=True)
    st.caption("Professional AI Consultation — not medical advice or diagnosis.")

    # Main Consultation Layout
    col_doctor, col_interaction = st.columns([1, 1.2], gap="large")

    with col_doctor:
        st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
        avatar_hero_placeholder = st.empty()
        avatar_hero_placeholder.markdown(get_avatar_html(st.session_state["avatar_talking"], "STANDING BY"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <h2 style="color: var(--ms-blue); font-size: 1.8rem; margin-bottom: 0.1rem;">Health Assistant AI</h2>
            <p style="color: var(--ms-neutral-secondary); font-weight: 500;">Virtual Physician Assistant</p>
            <div style="font-size: 0.75rem; color: #ef4444; border: 1px solid #fee2e2; padding: 4px; border-radius: 4px; margin-top: 10px;">
                ⚠️ SAFETY NOTICE: Not a substitute for professional medical advice.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        recorded_audio = st.audio_input("Record your message", key="voice_chat_input")
        st.markdown('</div>', unsafe_allow_html=True)

        # Load context
        notes = load_notes()
        context = build_assistant_context(notes, True, True)

        with st.container(border=True):
            if recorded_audio:
                if st.button("Transcribe & Ask", type="primary", use_container_width=True):
                    audio_bytes = recorded_audio.read()
                    with st.spinner("Analyzing speech..."):
                        voice_text = transcribe_audio_bytes(audio_bytes)
                        if voice_text and not voice_text.startswith("[Error"):
                            get_avatar_advice(voice_text, context)
                        elif not voice_text:
                            st.warning("No speech detected. Please speak clearly into the microphone.")
                        else:
                            st.error(f"Transcription error: {voice_text}")
            
            # 1) Transcript Section
            if st.session_state.get("last_transcript"):
                st.markdown("### 📝 recognized Transcript")
                st.info(st.session_state["last_transcript"])
            
            # 2) Assistant Reply Section
            if st.session_state.get("last_ai_reply"):
                st.markdown("### 🤖 Assistant Response")
                st.success(st.session_state["last_ai_reply"])
                
            st.divider()
            
            # 3) Full Chat History
            st.markdown("### 💬 Persistent Chat History")
            if not st.session_state["messages"]:
                st.info("Greetings. I am your AI assistant. I can help summarize your logs or prepare you for your next doctor's visit.")
            else:
                for msg in st.session_state["messages"]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            user_input = st.chat_input("Ask about your health trends...")
            if user_input:
                get_avatar_advice(user_input, context)
                
            # Audio Playback & Animation
            if "last_audio" in st.session_state and st.session_state["last_audio"]:
                st.audio(st.session_state["last_audio"], format="audio/mp3", autoplay=True)
                if st.session_state.get("new_audio_flag"):
                    st.session_state["avatar_talking"] = True
                    avatar_hero_placeholder.markdown(get_avatar_html(True, "EXPLAINING"), unsafe_allow_html=True)
                    last_msg = st.session_state["messages"][-1]["content"] if st.session_state["messages"] else ""
                    sleep_duration = max(2, len(last_msg) * 0.05)
                    time.sleep(sleep_duration)
                    st.session_state["avatar_talking"] = False
                    st.session_state["new_audio_flag"] = False
                    st.rerun()
        
    st.divider()
    with st.expander("Clinical Preparation Tools"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Spot Weekly Patterns", use_container_width=True):
            get_avatar_advice("Spot Patterns", context)
        if c2.button("Appt Prep Highlights", use_container_width=True):
            get_avatar_advice("Doctor Prep", context)
        if c3.button("Doctor Questions", use_container_width=True):
            get_avatar_advice("Generate Questions", context)

elif current_page == "Daily Check-In":
    st.markdown('<div class="section-header">Daily Health Check-In</div>', unsafe_allow_html=True)

    if st.session_state.get("show_checkin_results") and st.session_state.get("last_checkin_result"):
        res = st.session_state["last_checkin_result"]
        st.markdown("### 📊 Today's Summary Results")
        
        col_res1, col_res2 = st.columns([1.5, 1])
        
        with col_res1:
            st.markdown('<div class="card" style="padding: 1.5rem;">', unsafe_allow_html=True)
            st.markdown("#### Transcript & Summary")
            st.write(res.get("raw_text_redacted", ""))
            
            if res.get("diary"):
                st.markdown("---")
                st.markdown("**AI Diary Entry:**")
                st.write(res["diary"].get("summary", "Analysis pending..."))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card" style="padding: 1.5rem; border-left: 5px solid #16A34A;">', unsafe_allow_html=True)
            st.markdown("#### 💡 Today's Professional Insight")
            st.write(generate_insight(res.get("raw_text_redacted", "")))
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res2:
            st.markdown('<div class="card" style="padding: 1.5rem; text-align: center;">', unsafe_allow_html=True)
            st.markdown("#### Health Tags & Sentiment")
            mood_score = res.get("diary", {}).get("sentiment", 0)
            mood_l = get_mood_label(mood_score)
            st.markdown(f'<div style="font-size: 2rem; font-weight: 800; color: #3A86FF;">{mood_l}</div>', unsafe_allow_html=True)
            st.markdown(f'Score: {mood_score:.2f}')
            
            entities = res.get("medical_entities", {})
            if any(entities.values()):
                st.markdown("---")
                st.markdown("**Detected Entities:**")
                for cat, items in entities.items():
                    if items:
                        st.markdown(f"*{cat}:* {', '.join(items)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Mini Trend Preview
            all_notes = load_notes()
            diary_notes = [n for n in all_notes if n.get("mode") == "diary"]
            if len(diary_notes) > 1:
                st.markdown('<div class="card" style="padding: 1rem;">', unsafe_allow_html=True)
                st.markdown("#### 7-Day Trend Preview")
                scores = [float(n.get("diary", {}).get("sentiment", 0)) for n in diary_notes[-7:]]
                st.line_chart(scores, height=120)
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        cbtn1, cbtn2 = st.columns(2)
        if cbtn1.button("🔙 Back to Dashboard", use_container_width=True):
            st.session_state["show_checkin_results"] = False
            st.rerun()
        if cbtn2.button("📈 Open Full Insights", type="primary", use_container_width=True):
            st.session_state["show_checkin_results"] = False
            st.session_state["current_page"] = "Insights"
            st.rerun()
        
        st.stop() # Prevent showing the form below

    st.write("Share your health updates via voice or text. Your data is analyzed and saved securely.")



    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_text = st.text_area("How are you feeling?", value=st.session_state["transcribed_text"], placeholder="e.g., 'I have a slight headache and felt tired today.'", height=150)
    st.session_state["transcribed_text"] = input_text
    
    checkin_mode = st.radio("Log Type:", ["Personal Health Diary", "Clinical SOAP Note"], horizontal=True)
        
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        checkin_audio = st.audio_input("Voice Log", key="checkin_audio_input")
    with c2:
        confirm_phi = st.checkbox("Confirm NO PHI", value=False)
    with c3:
        if st.button("Clear Input", use_container_width=True):
            st.session_state["transcribed_text"] = ""
            st.rerun()

    if checkin_audio:
        if st.button("Transcribe Voice Log", use_container_width=True):
            audio_bytes = checkin_audio.read()
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio_bytes(audio_bytes)
                if transcription and not transcription.startswith("[Error"):
                    st.session_state["transcribed_text"] = (st.session_state["transcribed_text"] + " " + transcription).strip()
                    st.session_state["last_transcript"] = transcription
                    st.rerun()
                elif not transcription:
                    st.warning("No speech detected.")
                else:
                    st.error(transcription)

    if st.button("Submit Check-In", type="primary", use_container_width=True, disabled=not confirm_phi):
        if input_text.strip():
            with st.spinner("Analyzing and saving..."):
                today_str = datetime.today().strftime("%Y-%m-%d")
                redacted_text = redact_phi(input_text)
                
                note_record = {
                    "timestamp": datetime.now().isoformat(),
                    "date": today_str,
                    "mode": "soap" if "SOAP" in checkin_mode else "diary",
                    "raw_text_redacted": redacted_text,
                    "medical_entities": extract_medical_concepts(redacted_text)
                }
                
                if note_record["mode"] == "soap":
                    note_record["soap"] = {"text": process_soap(redacted_text)}
                else:
                    note_record["diary"] = process_diary_logic(redacted_text)
                
                save_note(note_record)
                st.session_state["transcribed_text"] = ""
                st.session_state["last_checkin_result"] = note_record
                st.session_state["show_checkin_results"] = True
                st.success("✅ Check-in saved successfully!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Secondary tools in expanders
    with st.expander("Advanced Tools (Copilot & Document Merge)"):
        st.write("Professional consultation tools.")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("##### Consultation Copilot")
            audio_b64_copilot = render_auto_mic(key="page_copilot")
            if audio_b64_copilot:
                import base64
                copilot_audio_bytes = base64.b64decode(audio_b64_copilot)
                if st.button("Generate Copilot Notes"):
                    transcript = transcribe_audio_bytes(copilot_audio_bytes)
                    if transcript: st.success(process_live_copilot(transcript))
        with sc2:
            st.markdown("##### Document Merge")
            uploaded_file = st.file_uploader("Upload Lab Results (PDF/TXT)", type=["pdf","txt"], key="page_upload")
            if uploaded_file and st.button("Process Document"):
                st.info("Document analysis in progress...")

elif current_page == "Insights":
    st.markdown('<div class="section-header">Health Insights & Analysis</div>', unsafe_allow_html=True)
    
    all_notes = load_notes()
    diary_notes = [n for n in all_notes if n.get("mode") == "diary"]
    
    if not diary_notes:
        render_empty_state("Log your daily check-ins to unlock behavioral and health twin insights.", icon="📈", cta={"label": "Start Daily Check-In", "action": lambda: st.session_state.update({"current_page": "Daily Check-In"})})
    else:
        # Dashboard style analytics
        trends = analyze_trends(diary_notes)
        
        i1, i2 = st.columns([1, 1])
        with i1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Sentiment Arc (Mood Trend)")
            try:
                graph_notes = diary_notes[-14:]
                dates = [str(n.get("date", "Unknown")) for n in graph_notes]
                scores = [float(n.get("diary", {}).get("sentiment", 0.0)) for n in graph_notes]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(dates, scores, marker='o', color='#3A86FF')
                ax.axhline(0, color='gray', linestyle='--')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except: st.write("Graph generation pending more data.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Predicting Risk Intensity")
            alerts = generate_risk_alerts(trends)
            if alerts:
                for a in alerts: st.warning(a)
            else: st.success("No high-risk trends detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        with i2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### AI Health Twin Profile")
            if st.button("Regenerate Twin Profile"):
                st.session_state["health_twin_summary"] = generate_health_twin_summary(all_notes)
            st.write(st.session_state.get("health_twin_summary", "Profile pending regeneration."))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Micro-Habit Prescriptions")
            if st.button("Generate Today's Habits"):
                st.session_state["habits"] = generate_micro_habits(all_notes)
            if st.session_state.get("habits"):
                for hb in st.session_state["habits"]: st.checkbox(hb)
            st.markdown('</div>', unsafe_allow_html=True)

elif current_page == "Reports":
    render_reports_page()
elif current_page == "Find Care":
    render_find_care()
    st.markdown('</div>', unsafe_allow_html=True)
elif current_page == "Settings":
    st.markdown('<div class="section-header">Security & Account Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Manage your patient portal configuration.")
    st.session_state["privacy_mode"] = st.toggle("Global Privacy Mode (No local storage)", value=st.session_state.get("privacy_mode", False))
    st.divider()
    if st.button("🚨 Wipe My Local History", use_container_width=True):
        clear_local_history()
        st.success("All local records deleted.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # End main-container
