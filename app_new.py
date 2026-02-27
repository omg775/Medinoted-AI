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
from scipy.io import wavfile
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
        --ms-blue: #00A4EF;
        --ms-yellow: #FFB900;
        --ms-gray: #737373;
        
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
            radial-gradient(at 0% 0%, rgba(0, 164, 239, 0.03) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(127, 186, 0, 0.03) 0px, transparent 40%);
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: var(--text-color);
    }
    
    [data-theme="light"] .stApp { 
        background-color: var(--ms-pure-white); 
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 164, 239, 0.05) 0px, transparent 40%),
            radial-gradient(at 100% 0%, rgba(127, 186, 0, 0.02) 0px, transparent 40%);
    }

    [data-theme="dark"] .stApp { 
        background-image: 
            url("https://www.transparenttextures.com/patterns/carbon-fibre.png"),
            radial-gradient(at 0% 0%, rgba(242, 80, 34, 0.06) 0px, transparent 40%),
            radial-gradient(at 100% 0%, rgba(127, 186, 0, 0.06) 0px, transparent 40%),
            radial-gradient(at 0% 100%, rgba(255, 185, 0, 0.06) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(0, 164, 239, 0.06) 0px, transparent 40%);
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

    [data-testid="stVerticalBlock"] > div:nth-child(1) .card { border-top: 6px solid var(--ms-orange); }
    [data-testid="stVerticalBlock"] > div:nth-child(2) .card { border-top: 6px solid var(--ms-green); }

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
        border-left: clamp(5px, 1.5vw, 8px) solid var(--ms-blue); 
        padding-left: clamp(1rem, 2.5vw, 2rem);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-theme="light"] .section-header { color: var(--ms-neutral-primary); }
    
    .kpi-tile { 
        background: var(--secondary-background-color); 
        padding: clamp(1.5rem, 4vw, 2.5rem); 
        border-radius: 2px; 
        border: 1px solid rgba(128, 128, 128, 0.15); 
    }
    [data-theme="light"] .kpi-tile { background: var(--ms-azure-white); border: 1px solid var(--ms-neutral-tertiary); }

    .kpi-value { font-size: clamp(2rem, 5vw, 3rem); font-weight: 800; line-height: 1; color: var(--text-color); }
    .kpi-label { font-size: clamp(0.85rem, 2vw, 1.1rem); color: var(--ms-neutral-secondary); font-weight: 700; margin-top: 1rem; text-transform: uppercase; }
    
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
        background: conic-gradient(from 180deg at 50% 50%, var(--ms-blue), var(--ms-green), var(--ms-yellow), var(--ms-orange), var(--ms-blue));
        filter: blur(2px);
        box-shadow: 0 0 clamp(40px, 10vw, 100px) rgba(0, 164, 239, 0.25);
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
        color: var(--ms-blue) !important;
        background: transparent !important;
        box-shadow: 0 clamp(-6px, -1vw, -10px) 0px var(--ms-blue) inset !important;
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

    .chat-bubble-assistant {
        background: var(--secondary-background-color) !important;
        border-left: 8px solid var(--ms-blue) !important;
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
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
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
    st.session_state["chat_history"] = []

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
        speech_config.speech_recognition_language="en-US"
        audio_config = speechsdk.AudioConfig(filename=tmp_path)
        
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        # We need a synchronous resolution here instead of async to avoid Streamlit event loop issues
        result = speech_recognizer.recognize_once()
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "" # Safe to return empty for ambient noise
        elif result.reason == speechsdk.ResultReason.Canceled:
            return "[Error: Speech recognition failed. Please check microphone format or Azure configuration.]"
    except Exception as e:
        return "[Error: Speech recognition failed. Please check microphone format or Azure configuration.]"

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
    for sym, count in trends.get("top_symptoms", []):
        if count >= 3:
            alerts.append(f"Persistence Alert: You reported '{sym}' {count} times recently. Consider consulting a clinician if it persists.")
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

def generate_chat_reply(user_message, context, chat_history):
    if detect_red_flags(user_message):
        return "SAFETY ALERT: Seek urgent medical help immediately by calling local emergency services or going to the nearest emergency room. If you are a minor, please talk to a trusted adult right away."
    
    system_prompt = f"""You are a supportive AI Care Assistant. You provide informational support ONLY.
CRITICAL SAFETY RULES:
1. NEVER diagnose. NEVER prescribe medication.
2. Use safe wording: "may", "could", "consider".
3. Keep responses EXTREMELY concise (max 25 words). No markdown.

Structure:
A) Empathetic acknowledgement.
B) One pattern or suggestion.
C) One follow-up question.

Context:
{context if context else "No recent logs available."}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-5:]: # last 5 to keep context short
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

def get_avatar_advice(user_message, context):
    """Unified function to get text and voice advice from OpenAI Assistant."""
    # 1. Generate text and voice advice
    with st.spinner("Avatar is synthesizing advice..."):
        # Add user message to local history
        st.session_state["chat_history"].append({"role": "user", "content": user_message})
        
        # Get Text Reply
        reply = generate_chat_reply(user_message, context, st.session_state["chat_history"])
        
        # Get Voice Advice
        audio_bytes = generate_tts_audio(reply)
        
        # Store Assistant Reply
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        
        # Auto-log to Analytics (Diary entry)
        entities = extract_medical_concepts(user_message)
        diary_data = process_diary_logic(user_message)
        
        # Check if message contains relevant health data or sentiment
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
            # Add a special tag for tracking assistant logs
            if "tags" not in note_record["diary"]: note_record["diary"]["tags"] = []
            if "Chat Insight" not in note_record["diary"]["tags"]:
                note_record["diary"]["tags"].append("Chat Insight")
            save_note(note_record)

        # Set Audio & Animation Flags
        if audio_bytes:
            st.session_state["last_audio"] = audio_bytes
            st.session_state["new_audio_flag"] = True
            
    st.rerun()

# -----------------------------------------------------------------------------
# Authentication UI
# -----------------------------------------------------------------------------
if not st.session_state["is_authenticated"]:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("""<div class="header-wrapper"><h1 class="main-title">Health Assistant AI</h1></div>""", unsafe_allow_html=True)
    
    col_spacer1, col_auth, col_spacer2 = st.columns([1, 2, 1])
    with col_auth:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        mode = st.radio("Select Mode:", ["Login", "Register"], horizontal=True)
        
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        
        if mode == "Register":
            password_confirm = st.text_input("Confirm Password", type="password")
            if st.button("Create Account", type="primary", use_container_width=True):
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
                        user_data_dir(safe_name) # Initialize dir
                        st.success("Account created successfully. Please login.")
        else:
            if st.button("Login", type="primary", use_container_width=True):
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
                            # Clear potentially stateful data on login
                            st.session_state["transcribed_text"] = ""
                            st.session_state["chat_history"] = []
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                    else:
                        st.error("Invalid credentials.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# Main UI Construction
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

with st.sidebar:
    st.markdown(f"**Logged in as: {st.session_state['username']}**")
    if st.button("Logout"):
        st.session_state["is_authenticated"] = False
        st.session_state["username"] = None
        st.session_state["chat_history"] = []
        st.session_state["transcribed_text"] = ""
        st.rerun()
    st.divider()
    
    st.markdown("### Secure Backend Status")
    missing_keys = check_azure_connections()
    
    if missing_keys:
        st.error(f"⚠️ Missing Keys in .env:\n{', '.join(missing_keys)}")
        st.warning("Please configure your `.env` file for AI features to work.")
    else:
        st.success("✅ Azure OpenAI Connected")
        st.success("✅ Speech Service Ready")
        
    st.divider()
    
    # Persistence Warning for Cloud
    if os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud" or os.environ.get("HOSTNAME") == "streamlit":
        st.warning("Note: In this cloud environment, data in JSON files will reset when the server restarts.")
    
    st.write("Tabs loaded successfully.")

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="privacy-banner">Synthetic/anonymized content only. Do NOT enter real personal data or PHI.</div>', unsafe_allow_html=True)

# Top Bar with Microsoft Logo
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("""
    <div class="header-wrapper">
        <svg class="ms-logo" viewBox="0 0 23 23" xmlns="http://www.w3.org/2000/svg">
            <path fill="#F25022" d="M1 1h10v10H1z"/>
            <path fill="#7FBA00" d="M12 1h10v10H12z"/>
            <path fill="#00A4EF" d="M1 12h10v10H1z"/>
            <path fill="#FFB900" d="M12 12h10v10H12z"/>
        </svg>
        <h1 class="main-title">Health Assistant AI</h1>
    </div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clinical Intelligence & Personal Health Log</div>', unsafe_allow_html=True)

status_class = f"status-{st.session_state['app_status'][0]}"
status_text = st.session_state["app_status"][1]
st.markdown(f'<span class="status-pill {status_class}">{status_text}</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Patient Data Capture",
    "Clinical Analytics",
    "AI Intelligence Hub"
])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Data Entry Portal</div>', unsafe_allow_html=True)
    
    input_text = st.text_area("Note Input", value=st.session_state["transcribed_text"], placeholder="Type or record your note here...", height=150, label_visibility="collapsed")
    st.session_state["transcribed_text"] = input_text
    
    mode = st.selectbox("Note Type:", ["Doctor SOAP Note", "Personal Health Diary"])
        
    st.markdown("---")
    
    # Action Button Group
    col_rec, col_proc, col_clr = st.columns([1, 1, 1])
    
    process_clicked = False
            
    # Privacy & Consent Card
    st.markdown('<div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px dashed #cbd5e1;">', unsafe_allow_html=True)
    st.markdown('<strong>Privacy & Consent</strong>', unsafe_allow_html=True)
    confirm_phi = st.checkbox("I confirm this input contains NO real personal data / PHI.", value=False)
    privacy_mode = st.checkbox("Privacy Mode (do not save anything locally)", value=False)
    st.markdown('<div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">We auto-redact emails/phones/DOB/name labels before saving.</div>', unsafe_allow_html=True)
    
    with st.expander("Privacy redaction preview"):
        redacted_preview = redact_phi(input_text)
        if input_text != redacted_preview:
            st.warning("PHI detected and redacted in current input.")
        st.write(redacted_preview if input_text else "No input to preview.")
    st.markdown('</div>', unsafe_allow_html=True)

    with col_proc:
        if st.button("Process Note", type="primary", use_container_width=True, disabled=not confirm_phi):
            process_clicked = True

    st.markdown('</div>', unsafe_allow_html=True)
    
    with col_rec:
        st.markdown("✅ Speech Service Connected<br>🎤 Recording Ready", unsafe_allow_html=True)
        # Use Streamlit's native browser microphone recorder widget
        audio_buffer = st.audio_input("Record voice note", label_visibility="collapsed")
        
        if audio_buffer is not None:
            # Using st.session_state to track processed state to avoid infinite transcribe looping
            audio_bytes = audio_buffer.getvalue()
            current_audio_id = hash(audio_bytes)
            
            if st.session_state.get("last_audio_id") != current_audio_id:
                st.session_state["app_status"] = ("ready", "Transcribing...")
                
                with st.spinner("Processing speech to text..."):
                    transcription = transcribe_audio_bytes(audio_bytes)
                    
                    if transcription:
                        # Append the transcript to the existing text
                        st.session_state["transcribed_text"] = (st.session_state["transcribed_text"] + " " + transcription).strip()
                    st.session_state["last_audio_id"] = current_audio_id
                
                st.session_state["app_status"] = ("ready", "Ready")
                st.rerun()
                
    with col_clr:
        if st.button("Clear Input", use_container_width=True):
            st.session_state["transcribed_text"] = ""
            if "last_audio_id" in st.session_state:
                del st.session_state["last_audio_id"]
            st.session_state["app_status"] = ("ready", "Ready")
            st.rerun()

    if process_clicked and input_text.strip():
        st.session_state["app_status"] = ("ready", "Processing")
        today_str = datetime.today().strftime("%Y-%m-%d")
        raw_clean = input_text.strip()
        redacted_text = redact_phi(raw_clean)
        
        note_record = {
            "timestamp": datetime.now().isoformat(),
            "date": today_str,
            "mode": "soap" if "SOAP" in mode else "diary",
            "raw_text_redacted": redacted_text,
            "medical_entities": extract_medical_concepts(redacted_text)
        }
        
        if note_record.get("mode") == "soap":
            with st.spinner("Generating SOAP note"):
                soap_txt = process_soap(redacted_text)
                note_record["soap"] = {"text": soap_txt}
                st.session_state["app_status"] = ("success", "Success")
                if not privacy_mode: st.session_state["app_status"] = ("success", "Saved")
                
                # Output Results Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">SOAP Protocol Output</div>', unsafe_allow_html=True)
                st.markdown(f"```text\n{soap_txt}\n```")
                st.download_button("Download SOAP Note", data=f"SOAP NOTE\n{today_str}\n\n{soap_txt}", file_name=f"soap_note_{today_str}.txt")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Quality Score Card
                score, checks, missing = calculate_quality_score(redacted_text, soap_txt)
                soap_data = note_record.get("soap", {})
                if isinstance(soap_data, dict):
                    soap_data["quality"] = {"score": score, "checks": checks}
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Clinical Protocol Validation</div>', unsafe_allow_html=True)
                st.progress(score / 100.0)
                st.markdown(f"**Quality Score: {score}%**")
                
                for text, passed in checks:
                    icon = "[OK]" if passed else "[WARN]"
                    st.write(icon + " " + (text.replace('✔', '').replace('⚠', '').strip() if isinstance(text, str) else str(text)))
                
                if missing:
                    st.warning(f"**Missing:** {', '.join(missing)}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Continuity
                all_notes = load_notes()
                related = find_related_diary_entries(note_record["medical_entities"], all_notes)
                if related:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">Longitudinal Patient Insights</div>', unsafe_allow_html=True)
                    for r in related:
                        st.info(f"**{r['date']}** | Tags: {', '.join(r.get('diary',{}).get('tags',[]))} | _'{r['raw_text_redacted'][:100]}...'_")
                    st.markdown('</div>', unsafe_allow_html=True)

        else:
            with st.spinner("Analyzing diary..."):
                d_res = process_diary_logic(redacted_text)
                note_record["diary"] = d_res
                st.session_state["app_status"] = ("success", "Processed")
                if not privacy_mode: st.session_state["app_status"] = ("success", "Saved")
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Behavioral Health Synthesis</div>', unsafe_allow_html=True)
                tags = d_res.get('tags', [])
                st.markdown(f"**Tags:** {', '.join(tags) if isinstance(tags, list) and tags else 'None'}  \n**Sentiment:** `{d_res.get('sentiment', 0.0)}`")
                suggestions = d_res.get('suggestions', [])
                st.markdown(f"**AI Suggestions:**  \n{chr(10).join(suggestions) if isinstance(suggestions, list) and suggestions else '• None at this time.'}")
                st.markdown('</div>', unsafe_allow_html=True)

        # Medical Entity Chips Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Semantic Entity Extraction</div>', unsafe_allow_html=True)
        st.markdown(render_chips(note_record["medical_entities"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if not privacy_mode:
            save_note(note_record)

with tab2:
    all_notes = load_notes()
    diary_notes = [n for n in all_notes if n.get("mode") == "diary"]
    
    # KPI Tiles
    days_7_ago = datetime.today() - timedelta(days=7)
    recent_count = sum(1 for n in all_notes if datetime.strptime(n.get("date", "1970-01-01"), "%Y-%m-%d") >= days_7_ago)
    
    trends = analyze_trends(diary_notes)
    avg_mood = trends.get("sentiment_avg", 0)
    top_symp = trends.get("top_symptoms", [("None", 0)])[0][0] if trends.get("top_symptoms") else "None"
    
    last_timestamp = "N/A"
    if all_notes:
        last_dt = datetime.fromisoformat(all_notes[-1].get("timestamp", datetime.now().isoformat()))
        last_timestamp = last_dt.strftime("%b %d, %H:%M")

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{recent_count}</div><div class="kpi-label">Entries (7d)</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{avg_mood:.2f}</div><div class="kpi-label">Avg Mood (7 notes)</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-tile"><div class="kpi-value">{top_symp.title()}</div><div class="kpi-label">Top Symptom</div></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi-tile"><div class="kpi-value" style="font-size: 1.4rem; padding-top: 0.4rem;">{last_timestamp}</div><div class="kpi-label">Last Logged</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 1])
    
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Historical Data Streams</div>', unsafe_allow_html=True)
        reversed_notes = list(reversed(all_notes))
        for n in reversed_notes[:5]:
            if not isinstance(n, dict): continue
            badge_color = "#e0f2fe" if n.get('mode') == 'soap' else "#fef08a"
            badge_text = "SOAP" if n.get('mode') == 'soap' else "DIARY"
            redacted = n.get('raw_text_redacted', '')
            if not isinstance(redacted, str): redacted = str(redacted)
            st.markdown(f"""
            <div style="margin-bottom: 0.8rem; padding-bottom: 0.8rem; border-bottom: 1px solid #f1f5f9;">
                <span style="font-size: 0.8rem; color: #64748b; margin-right: 10px;">{n.get('date', 'Unknown Date')}</span>
                <span style="background-color: {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold;">{badge_text}</span>
                <div style="font-size: 0.95rem; margin-top: 5px; color: #334155;">{redacted[:80]}...</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Predictive Risk Intelligence</div>', unsafe_allow_html=True)
        alerts = generate_risk_alerts(trends)
        if alerts:
            for a in alerts: st.warning(a)
        else:
            st.success("No high-risk trends detected in recent history.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Behavioral Sentiment Mapping</div>', unsafe_allow_html=True)
        if isinstance(diary_notes, list) and len(diary_notes) > 1:
            try:
                # Type safe access for graphing
                graph_notes = diary_notes[-10:]
                dates = [str(n.get("date", "Unknown")) for n in graph_notes if isinstance(n, dict)]
                scores = []
                for n in graph_notes:
                    if isinstance(n, dict):
                        d_info = n.get("diary", {})
                        if isinstance(d_info, dict):
                            scores.append(float(d_info.get("sentiment", 0.0)))
                        else:
                            scores.append(0.0)
                    else:
                        scores.append(0.0)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(dates, scores, marker='o', color='#0ea5e9', linewidth=2)
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                ax.set_ylim(-1.1, 1.1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xticks(rotation=45, ha='right')
                fig.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not render graph: {e}")
        else:
            st.info("Log more Diary entries to generate sentiment arcs over time.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Symptomatic Trend Analysis</div>', unsafe_allow_html=True)
        if trends and isinstance(trends, dict):
            top_syms_dashboard = trends.get("top_symptoms", [])
            if isinstance(top_syms_dashboard, list) and top_syms_dashboard:
                for item in top_syms_dashboard:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        sym_name = str(item[0])
                        sym_count = int(item[1])
                        st.markdown(f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="font-weight: 500;">{sym_name.title()}</span><span style="color: #64748b;">{sym_count} occurrences</span></div>', unsafe_allow_html=True)
            else:
                st.write("No recurring symptoms detected yet.")
        else:
            st.write("No recurring symptoms detected yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AI Care Assistant Tab (Chat UI)
# -----------------------------------------------------------------------------
with tab3:
    st.caption("AI Consultation Hub — not medical advice or diagnosis.")

    # Main Consultation Layout
    col_doctor, col_interaction = st.columns([1, 1.2], gap="large")

    with col_doctor:
        # 1. Standing Doctor Avatar (Left Column)
        st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
        avatar_hero_placeholder = st.empty()
        avatar_hero_placeholder.markdown(get_avatar_html(st.session_state["avatar_talking"], "STANDING BY"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <h2 style="color: var(--ms-blue); font-size: 1.8rem; margin-bottom: 0.1rem;">Dr. Microsoft 3D</h2>
            <p style="color: var(--ms-neutral-secondary); font-weight: 500;">Virtual Physician Assistant</p>
        </div>
        """, unsafe_allow_html=True)

    with col_interaction:
        # 2. Voice & Interaction (Right Column)
        st.markdown("### Talk with your Doctor")
        if HAS_MIC_RECORDER:
            hub_audio = mic_recorder(
                start_prompt="Start Recording Voice",
                stop_prompt="Stop Recording",
                key='hub_recorder'
            )
        else:
            st.error("Microphone library missing.")
            hub_audio = None

        # Load context
        notes = load_notes()
        use_diary = st.session_state.get("use_diary_hub", True)
        use_soap = st.session_state.get("use_soap_hub", True)
        context = build_assistant_context(notes, use_soap, use_diary)

        st.markdown('<div class="card" style="margin-top: 1rem; padding: 1rem;">', unsafe_allow_html=True)
        
        # Process Voice Input
        if hub_audio:
            audio_id = hash(hub_audio['bytes'])
            if st.session_state["last_processed_audio"] != audio_id:
                st.session_state["last_processed_audio"] = audio_id
                voice_text = transcribe_audio(hub_audio['bytes'])
                if voice_text:
                    get_avatar_advice(voice_text, context)
        
        # Display History or Welcome
        if not st.session_state["chat_history"]:
            st.info("I'm grounded in your data (Diary & SOAP notes). Ask me anything about your trends.")
        
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Input handling
        user_input = st.chat_input("Message your assistant...")
        
        # Quick Prompts
        st.write("") # Spacer
        qp1, qp2, qp3 = st.columns(3)
        prompt_clicked = None
        if qp1.button("Diary Summary", use_container_width=True): prompt_clicked = "Summarize my recent diary entries"
        if qp2.button("Spot Patterns", use_container_width=True): prompt_clicked = "What health patterns do you see?"
        if qp3.button("Doctor Prep", use_container_width=True): prompt_clicked = "Help me prepare for my next clinic visit"

        if prompt_clicked:
            user_input = prompt_clicked

        if user_input:
            get_avatar_advice(user_input, context)
        
        # Audio Playback
        if "last_audio" in st.session_state and st.session_state["last_audio"]:
            st.audio(st.session_state["last_audio"], format="audio/mp3", autoplay=True)
            if st.session_state.get("new_audio_flag"):
                st.session_state["avatar_talking"] = True
                avatar_hero_placeholder.markdown(get_avatar_html(True, "EXPLAINING"), unsafe_allow_html=True)
                last_msg = st.session_state["chat_history"][-1]["content"] if st.session_state["chat_history"] else ""
                sleep_duration = max(2, len(last_msg) * 0.05)
                time.sleep(sleep_duration)
                st.session_state["avatar_talking"] = False
                st.session_state["new_audio_flag"] = False
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Global Settings & Technical Logs (Footer)
    st.divider()
    coll1, coll2, coll3 = st.columns([1,1,1])
    with coll1:
        if st.button("New Consultation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["last_processed_audio"] = None
            st.rerun()
    with coll2:
        st.session_state["use_diary_hub"] = st.toggle("Include Diary Logs", value=True)
    with coll3:
        st.session_state["use_soap_hub"] = st.toggle("Include SOAP Notes", value=True)

    with st.expander("Technical Insight: Knowledge Context"):
        st.code(context if context else "No context loaded.")

st.markdown('</div>', unsafe_allow_html=True) # End main-container

