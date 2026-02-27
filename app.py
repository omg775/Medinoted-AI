# REQUIRED TO INSTALL:
# pip install streamlit matplotlib vaderSentiment sounddevice scipy openai
# 
# OPTIONAL (But Highly Recommended for full functionality):
# pip install openai-whisper
# pip install SpeechRecognition
# pip install spacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

import bcrypt
from dotenv import load_dotenv
import streamlit as st
import sqlite3
import os
import json
import re
import io
import logging
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Set
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io import wavfile
from openai import OpenAI, AzureOpenAI
import time
import textwrap

# Load environment variables from .env file, overriding any cached ones to allow hot-reloading
load_dotenv(override=True)
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False

try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
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

try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_AZURE_SPEECH = True
except ImportError:
    HAS_AZURE_SPEECH = False


# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Health Assistant AI", page_icon="https://www.microsoft.com/favicon.ico", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root {
        --doctor-blue: #2b7cff;
        --patient-green: #27ae60;
        --ai-purple: #9b59b6;
        --alert-orange: #f39c12;
        --glass-bg: rgba(255, 255, 255, 0.75);
        --glass-border: rgba(255, 255, 255, 0.4);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }
    
    /* Base App Styling (Gradients are injected dynamically per role later) */
    .stApp {
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Sidebar Styling - Dark Professional Tone */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: white !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }

    /* Glass Card System */
    .card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        box-shadow: var(--glass-shadow);
        padding: 2rem;
        margin-bottom: 2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.12);
    }
    
    /* Role Specific Card Borders */
    .card-doctor {
        border-left: 6px solid var(--doctor-blue);
    }
    
    /* GLOBAL RESPONSIVE SYSTEM */
    @media screen and (max-width: 900px) {
        /* Tablet adjustments */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            max-width: 100% !important;
            min-width: 100% !important;
            display: block !important;
            margin-bottom: 1rem;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            flex-direction: column !important;
        }
    }
    
    @media screen and (max-width: 600px) {
        /* Mobile phone adjustments */
        
        /* Forces all layout columns to stack vertically, preventing horizontal squishing */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            max-width: 100% !important;
            display: block !important;
            margin-bottom: 1.5rem;
        }
        
        /* Forces columns wrapper to strictly stack vertically */
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            flex-wrap: wrap !important;
            padding: 0 !important;
            gap: 0 !important;
        }
        
        /* Buttons go full width and grow for touch targets */
        div.stButton > button {
            width: 100% !important;
            min-height: 48px !important;
            margin-top: 0.5rem;
        }
        
        /* Inputs span full width */
        div[data-baseweb="input"], 
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Dashboard/Chart Containers shrink-wrap */
        .card {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 12px;
        }
        
        /* Adjust global typography scales */
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
        
        /* Simplify Avatar Box on Mobile Login */
        .avatar-box {
            width: 85% !important;
            max-width: 250px !important;
            margin: 0 auto 1.5rem auto !important;
        }
        
        /* AI Chat inputs stay fixed and readable */
        .stChatInput {
            padding-bottom: 1rem !important;
        }
        .stChatMessage {
            padding: 0.75rem !important;
        }
        
        /* Shrink Sidebar padding to prevent overflow */
        [data-testid="stSidebar"] {
            padding: 1rem !important;
            width: 85vw !important; /* ensure it doesn't overrun entire screen */
        }
    }
    
    .card-patient {
        border-left: 6px solid var(--patient-green);
    }
    
    .card-ai {
        border-left: 6px solid var(--ai-purple);
    }

    /* Section Headers */
    .section-header {
        font-weight: 800;
        font-size: 1.4rem;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Button Modernization */
    .stButton > button {
        border-radius: 12px !important;
        height: 50px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2b7cff 0%, #0052cc 100%) !important;
        color: white !important;
    }

    /* KPI Tiles inside Dashboard */
    .kpi-tile {
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    /* Chat UI Polish */
    .chat-bubble-assistant {
        background: rgba(243, 244, 246, 0.8) !important;
        border-left: 5px solid var(--ai-purple) !important;
        border-radius: 12px 12px 12px 0px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    
    .chat-bubble-user {
        background: rgba(43, 124, 255, 0.1) !important;
        border-right: 5px solid var(--doctor-blue) !important;
        border-radius: 12px 12px 0px 12px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        text-align: right !important;
        color: #1f2937 !important;
        font-weight: 500 !important;
    }

    /* Tabs Styling Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 1rem 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1e293b !important;
        border-bottom-color: #1e293b !important;
    }

    /* Hero Banner Header */
    .hero-banner {
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.95)), url('https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        box-shadow: var(--glass-shadow);
        border: 1px solid var(--glass-border);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 1rem;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: #1e293b;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    st.session_state["notes_db"] = load_notes()
if "app_status" not in st.session_state:
    st.session_state["app_status"] = ("ready", "Ready")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "avatar_talking" not in st.session_state:
    st.session_state["avatar_talking"] = False
if "last_processed_audio" not in st.session_state:
    st.session_state["last_processed_audio"] = None
if "distress_alert" not in st.session_state:
    st.session_state["distress_alert"] = False
if "active_tab_trigger" not in st.session_state:
    st.session_state["active_tab_trigger"] = None

# -----------------------------------------------------------------------------
# Core Storage Functions (Local JSON Profile)
# -----------------------------------------------------------------------------
PROFILE_PATH = os.path.join("data", "user_profile.json")

def load_user_profile():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(PROFILE_PATH):
        return {"streak_count": 0, "last_log_date": None, "logs": [], "privacy_mode": True}
    try:
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"streak_count": 0, "last_log_date": None, "logs": [], "privacy_mode": True}

def save_user_profile(profile_data):
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile_data, f, indent=4)

def load_notes():
    profile = load_user_profile()
    return profile.get("logs", [])

def save_note(note):
    profile = load_user_profile()
    today_str = datetime.today().strftime("%Y-%m-%d")
    
    note["timestamp"] = note.get("timestamp", datetime.now().isoformat())
    note["date"] = note.get("date", today_str)
    
    # Optional Privacy Masking (Don't save raw text to disk if privacy is on)
    if profile.get("privacy_mode", True):
        # We might still want it in memory for the current session render, 
        # so we strip it off the persisted copy.
        persisted_note = dict(note)
        if "raw_text_redacted" in persisted_note:
            persisted_note["raw_text_redacted"] = "[HIDDEN BY PRIVACY MODE]"
        if "soap" in persisted_note and "text" in persisted_note["soap"]:
            persisted_note["soap"]["text"] = "[HIDDEN BY PRIVACY MODE]"
        profile["logs"].append(persisted_note)
    else:
        profile["logs"].append(note)
        
    # Streak Logic Calculation
    last_log = profile.get("last_log_date")
    if last_log != today_str:
        if last_log:
            try:
                last_date = datetime.strptime(last_log, "%Y-%m-%d")
                today_date = datetime.strptime(today_str, "%Y-%m-%d")
                delta = (today_date - last_date).days
                if delta == 1:
                    profile["streak_count"] += 1
                elif delta > 1:
                    profile["streak_count"] = 1 # Reset if skipped a day
            except ValueError:
                 profile["streak_count"] = 1
        else:
            profile["streak_count"] = 1 # First time logging
            
        profile["last_log_date"] = today_str
        
    save_user_profile(profile)
    # Maintain in-memory copy for immediate UI sync
    if "notes_db" not in st.session_state:
        st.session_state["notes_db"] = []
    st.session_state["notes_db"] = profile["logs"]

def clear_local_history():
    st.session_state["notes_db"] = []
    st.session_state["chat_history"] = []
    st.session_state["transcribed_text"] = ""
    # Wipe JSON
    if os.path.exists(PROFILE_PATH):
        try:
            os.remove(PROFILE_PATH)
        except Exception:
            pass
    # Reset default file
    save_user_profile({"streak_count": 0, "last_log_date": None, "logs": [], "privacy_mode": True})

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

@st.cache_resource(show_spinner="Loading Whisper Model...")
def load_whisper_model():
    if HAS_WHISPER:
        return whisper.load_model("base")
    return None

def transcribe_audio(audio_bytes):
    if not audio_bytes or len(audio_bytes) < 10000: # ~0.1s of audio depending on bitrate
        return ""
    
    # Priority 1: Azure Speech STT
    if HAS_AZURE_SPEECH and os.environ.get("AZURE_SPEECH_KEY"):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            transcript = transcribe_audio_azure(audio_file_path=tmp_path)
            if transcript and not transcript.startswith("⚠️") and "No speech" not in transcript and "canceled" not in transcript:
                return transcript.strip()
            elif transcript and transcript.startswith("⚠️"):
                 st.warning(transcript)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Priority 2: OpenAI Cloud STT
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY") != "your_api_key_here":
        try:
            client = OpenAI()
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = "audio.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_buffer
            ).text
            return transcript.strip()
        except Exception as e:
            # Check if this is a short audio error, if so just silently ignore
            error_str = str(e)
            if "too short" in error_str.lower() or "0.1 seconds" in error_str.lower() or "401" in error_str:
                return ""
            st.warning("⚠️ **Cloud STT failed.** Falling back to local methods. Error:")
            st.code(error_str)

    # Fallback to Local Methods
    audio_buffer = io.BytesIO(audio_bytes)
    if HAS_WHISPER:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            model = load_whisper_model()
            if model:
                result = model.transcribe(tmp_path, fp16=False)
                return result["text"].strip()
            return "[Error: Whisper model failed to load.]"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif HAS_SR:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_buffer) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data)
            except:
                return "Google Speech Recognition could not understand audio."
    else:
        return "[Error: STT not configured. Provide OpenAI API Key or install local Whisper.]"

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
    if not diary_notes:
        return {}
        
    total_sentiment = 0
    symptoms = []
    
    for n in diary_notes:
        d = n.get("diary", {})
        total_sentiment += d.get("sentiment", 0.0)
        symptoms.extend(d.get("tags", []))
        
    avg = total_sentiment / len(diary_notes)
    
    # Count symptoms manually
    symp_counts = {}
    for s in symptoms:
        symp_counts[s] = symp_counts.get(s, 0) + 1
    
    # sort by count
    top_symps = sorted(symp_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "sentiment_avg": avg,
        "top_symptoms": top_symps
    }

def generate_risk_alerts(trends):
    alerts = []
    if trends.get("sentiment_avg", 0) < -0.3:
        alerts.append("Mood Alert: Your sentiment has been trending downwards recently. Consider practicing self-care or speaking with someone you trust.")
    for sym, count in trends.get("top_symptoms", []):
        if count >= 3:
            alerts.append(f"Persistence Alert: You reported '{sym}' {count} times recently. Consider consulting a clinician if it persists.")
    return alerts

# -----------------------------------------------------------------------------
# Processors
# -----------------------------------------------------------------------------
def detect_distress_keywords(text):
    distress_words = ["severe pain", "can't breathe", "fainted", "unconscious", "accident", "bleeding heavily", "chest pain", "suicide", "kill myself", "heart attack"]
    text_lower = text.lower()
    return any(word in text_lower for word in distress_words)

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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    # Use the Azure OpenAI wrapper
    reply = chat_reply(messages)
    
    if reply.startswith("⚠️ Error connecting") or reply.startswith("⚠️ Azure"):
        return f"SOAP Event Failed: {reply}"
    return reply

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
# Azure OpenAI Conversational Chatbot
# -----------------------------------------------------------------------------
def chat_reply(messages):
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat").strip()
    
    # The Python SDK expects the base URL. If the user accidentally includes /openai/ at the end, 
    # it causes a 404 error because the SDK appends /openai/deployments/... itself.
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]
    if endpoint.endswith("/openai"):
        endpoint = endpoint[:-7]
    
    if not (endpoint and api_key):
        return "⚠️ Azure OpenAI credentials missing. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env."
    
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg and "Resource not found" in error_msg:
            return f"⚠️ **Azure Error 404**: The deployment '{deployment}' could not be found at the provided endpoint. Please verify `AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_ENDPOINT` exactly match your Azure AI Studio setup."
        return f"Error connecting to Azure OpenAI: {error_msg}"

def transcribe_audio_azure(audio_file_path=None, use_local_mic=False):
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    service_region = os.environ.get("AZURE_SPEECH_REGION")
    
    if not (speech_key and service_region):
        return "⚠️ Azure Speech credentials missing. Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."
        
    if not HAS_AZURE_SPEECH:
        return "⚠️ azure-cognitiveservices-speech library is not installed."

    try:
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        
        if use_local_mic:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            st.info("Listening... Speak into your microphone.")
            result = speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return "No speech could be recognized."
            elif result.reason == speechsdk.ResultReason.Canceled:
                return f"Speech Recognition canceled: {result.cancellation_details.reason}"
                
        elif audio_file_path:
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            result = speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return "No speech could be recognized in the file."
            elif result.reason == speechsdk.ResultReason.Canceled:
                return f"Speech Recognition canceled: {result.cancellation_details.reason}"
                
        return None
    except Exception as e:
        return f"Error with Azure Speech STT: {e}"

def generate_tts_audio_azure(text):
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    service_region = os.environ.get("AZURE_SPEECH_REGION")
    
    if not (speech_key and service_region):
        return None
        
    if not HAS_AZURE_SPEECH:
        return None
        
    try:
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        # Choose a friendly, clinical voice
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        # We need the output as an audio stream (bytes), so we disable default speaker output
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        result = speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            st.error(f"TTS Synthesis Failed: {result.reason}")
            return None
    except Exception as e:
        st.error(f"Error with Azure Speech TTS: {e}")
        return None

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


def get_avatar_advice(user_message, context):
    """Unified function to get text and voice advice from OpenAI Assistant."""
    # 1. Generate text and voice advice
    with st.spinner("Avatar is synthesizing advice..."):
        # Add user message to local history
        st.session_state["chat_history"].append({"role": "user", "content": user_message})
        
        # Get Text Reply
        if detect_red_flags(user_message):
            reply = "SAFETY ALERT: Seek urgent medical help immediately by calling local emergency services or going to the nearest emergency room."
        else:
            system_prompt = f"""You are a supportive AI Care Assistant. You provide informational support ONLY.
CRITICAL SAFETY RULES:
1. NEVER diagnose. NEVER prescribe medication.
2. Use safe wording: "may", "could", "consider".
3. Keep responses EXTREMELY concise, friendly, and clinical. No markdown.

Context:
{context if context else "No recent logs available."}"""
            messages = [{"role": "system", "content": system_prompt}]
            # Chat history already has the user_message appended
            for msg in st.session_state["chat_history"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
            reply = chat_reply(messages)
        
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

        # We no longer auto-generate/play TTS here; user requests it via button.
    st.rerun()

# -----------------------------------------------------------------------------
# Main UI Construction
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏥 Health Intelligence")
    st.divider()
    st.markdown("### Privacy Session")
    if st.button("🗑️ Clear Session Data", help="Wipes all in-memory structured data and history.", use_container_width=True):
        clear_local_history()
        st.session_state["transcribed_text"] = ""
        st.rerun()
    st.divider()
    # API Key Status
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_SPEECH_KEY"):
        st.success("Azure AI Features Active")
    elif os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY") != "your_api_key_here":
        st.success("Legacy OpenAI Features Active")
    else:
        st.error("AI keys not configured in .env file.")

app_bg = "linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%)"

st.markdown(f"""
<style>
    .stApp {{
        background: {app_bg} !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="privacy-banner">🔒 Demo uses synthetic/anonymized content. Not a diagnostic tool.</div>', unsafe_allow_html=True)

# Hero Banner
st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">Health Intelligence Assistant</div>
        <div class="hero-subtitle">Voice-Powered Documentation & Personal Health Insights</div>
    </div>
""", unsafe_allow_html=True)

tab_health, tab_insights, tab_travel = st.tabs([
    "💬 Log Health",
    "📊 Insights",
    "✈️ Travel Care"
])

card_class = "card-patient"

with tab_health:
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h2 style="font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; font-size: 2.2rem;">How are you feeling today?</h2>
            <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">Speak or type your symptoms, mood, or general health updates. We'll summarize your entry and extract key insights.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Single Input Area
    health_input = st.text_area(
        "Log Health", 
        value=st.session_state.get("transcribed_text", ""), 
        placeholder="e.g. I woke up with a slight headache and feeling a bit tired. I took some ibuprofen...", 
        height=140, 
        label_visibility="collapsed"
    )
    st.session_state["transcribed_text"] = health_input
    
    st.markdown("<br/>", unsafe_allow_html=True)
    col_mic, col_space, col_submit = st.columns([1, 1.5, 1])
    
    analyze_clicked = False
    with col_submit:
        if st.button("Analyze Entry ✨", type="primary", use_container_width=True):
            analyze_clicked = True
            
    with col_mic:
        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] button {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🎤 Dictate", use_container_width=True):
             with st.spinner("Listening natively via Azure..."):
                 voice_text = transcribe_audio_azure(use_local_mic=True)
                 if voice_text and not voice_text.startswith("⚠️"):
                     st.session_state["transcribed_text"] = (st.session_state.get("transcribed_text", "") + " " + voice_text).strip()
                     st.success("Dictation captured. Click Analyze Entry.")
                     st.rerun()
                 else:
                     st.warning(voice_text)
    
    # Progressive Disclosure: Advanced Options
    with st.expander("⚙️ Advanced Options & Privacy"):
        st.markdown('<div style="font-size: 0.9rem; color: #475569; margin-bottom: 0.5rem;">Configure how your data is processed safely.</div>', unsafe_allow_html=True)
        confirm_phi = st.checkbox("I confirm this contains NO real personal data or PHI.", value=True, help="We auto-redact specific identifiers before processing.")
        if st.button("Clear Input Text"):
            st.session_state["transcribed_text"] = ""
            st.rerun()

    # Results & Chat-Style Assistant Response
    if analyze_clicked and health_input.strip():
        if not confirm_phi:
            st.error("Please confirm the data contains no PHI in the Advanced Options before proceeding.")
        else:
            with st.spinner("Processing your entry..."):
                today_str = datetime.today().strftime("%Y-%m-%d")
                redacted_text = redact_phi(health_input.strip())
                
                d_res = process_diary_logic(redacted_text)
                
                # Check for distress
                if detect_distress_keywords(redacted_text):
                    st.session_state["distress_alert"] = True
                else:
                    st.session_state["distress_alert"] = False
                    
                note_record = {
                    "timestamp": datetime.now().isoformat(),
                    "date": today_str,
                    "mode": "diary",
                    "raw_text_redacted": redacted_text,
                    "medical_entities": extract_medical_concepts(redacted_text),
                    "diary": d_res
                }
                save_note(note_record)
                
                st.markdown("<hr style='margin: 3rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
                
                # Chat-style output
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown("### Health Synthesis Summary")
                    st.markdown("Here is my analysis of your entry:")
                    
                    # Styled Tags
                    tags = d_res.get('tags', [])
                    if tags:
                        tag_html = "".join([f'<span style="background:#e0e7ff; color:#4338ca; padding:4px 10px; border-radius:12px; font-size:0.85rem; margin-right:6px; font-weight:600;">{t}</span>' for t in tags])
                        st.markdown(f'<div style="margin: 10px 0;">{tag_html}</div>', unsafe_allow_html=True)
                        
                    st.markdown(f"**Sentiment Score:** `{d_res.get('sentiment', 0.0)}`")
                    
                    suggestions = d_res.get('suggestions', [])
                    if suggestions:
                        st.markdown("**Suggestions:**")
                        for s in suggestions: 
                            st.markdown(f"• {s}")
                            
                    st.info("💡 **Disclaimer:** This tool provides broad informational summaries, not medical diagnoses.")
                    
                    # Contextual Travel Care Action
                    if st.session_state.get("distress_alert"):
                        st.error("⚠️ **Notice:** Some symptoms you mentioned may require timely professional attention.")
                        if st.button("🚨 Find Care Nearby", use_container_width=True, type="primary"):
                            st.session_state["active_tab_trigger"] = "Travel Care"
                            st.rerun()
                            
                    st.markdown("<br/>", unsafe_allow_html=True)
                    
                    # Progressive Disclosure: Clinical Actions
                    with st.expander("📄 Convert to Clinical SOAP Note"):
                        st.markdown("Use this to generate a structured clinical document for your personal records.")
                        if st.button("Generate SOAP Format"):
                            with st.spinner("Structuring into Subjective, Objective, Assessment, Plan..."):
                                soap_txt = process_soap(redacted_text)
                                st.markdown(f"```text\n{soap_txt}\n```")
                                st.download_button("📥 Download as TXT", data=f"SOAP NOTE\n{today_str}\n\n{soap_txt}", file_name=f"clinical_note_{today_str}.txt")

with tab_insights:
    all_notes = load_notes()
    diary_notes = [n for n in all_notes if n.get("mode") == "diary"]
    soap_notes = [n for n in all_notes if n.get("mode") == "soap"]

    if not all_notes:
        st.info("No data logged yet. Add some Diary entries or SOAP notes to see insights.")
    else:
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
            st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📈 Historical Activity Stream</div>', unsafe_allow_html=True)
            reversed_notes = list(reversed(all_notes))
            for n in reversed_notes[:6]:
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

        with colB:
            st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🗺️ Behavioral Trends</div>', unsafe_allow_html=True)
            if isinstance(diary_notes, list) and len(diary_notes) > 1:
                try:
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

            st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔍 Symptomatic Frequency</div>', unsafe_allow_html=True)
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

st.markdown('</div>', unsafe_allow_html=True) # End main-container

with tab_travel:
    st.markdown(f'<div class="card {card_class}">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">✈️ Care Navigation & Travel Health</div>', unsafe_allow_html=True)
    
    st.info("💡 **Disclaimer:** This is a prototype for care navigation. It is not a medical diagnosis tool, nor does it guarantee insurance coverage. Mention of facilities does not constitute endorsement.")
    
    if st.session_state.get("distress_alert"):
        st.error("🚨 **System Notice:** Based on recent diary/SOAP entries, you mentioned symptoms that may require timely professional attention. Use the search below to find nearby facilities.")
        if st.button("Dismiss Alert"):
            st.session_state["distress_alert"] = False
            st.rerun()

    st.markdown("### Find Care Near Me")
    st.markdown('<div style="color: #64748b; margin-bottom: 1rem;">Search our simulated partner hospital network across the globe.</div>', unsafe_allow_html=True)
        
    col_search1, col_search2 = st.columns(2)
    with col_search1:
        search_country = st.text_input("Country", placeholder="e.g. France, Japan...")
    with col_search2:
        search_city = st.text_input("City", placeholder="e.g. Paris, Tokyo...")
        
    if st.button("Search Facilities", type="primary", use_container_width=True):
        st.markdown("---")
        try:
            with open("data/partner_facilities.json", "r") as f:
                facilities = json.load(f)
                
            results = facilities
            if search_country:
                results = [f for f in results if search_country.lower() in f.get("country", "").lower()]
            if search_city:
                results = [f for f in results if search_city.lower() in f.get("city", "").lower()]
                
            if not results:
                st.warning("No facilities found matching your search. Try broadening your criteria.")
            else:
                st.success(f"Found {len(results)} facilities in our network.")
                
                # Render map if lat/lon available
                map_data = []
                for fac in results:
                    if "lat" in fac and "lon" in fac:
                        map_data.append({"LAT": fac["lat"], "LON": fac["lon"], "name": fac["facility_name"]})
                
                if map_data:
                    import pandas as pd
                    import pydeck as pdk
                    df = pd.DataFrame(map_data)
                    
                    view_state = pdk.ViewState(
                        latitude=47.1625,
                        longitude=19.5033,
                        zoom=3,
                        pitch=0
                    )
                    
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=df,
                        get_position="[LON, LAT]",
                        get_color="[220, 38, 38, 160]",
                        get_radius=150000,
                        pickable=True
                    )
                    
                    st.pydeck_chart(pdk.Deck(
                        map_style=None,
                        initial_view_state=view_state,
                        layers=[layer],
                        tooltip={"text": "{name}"}
                    ))
                
                # Render list
                for fac in results:
                    st.markdown(f"""
                    <div style="background-color: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <h4 style="margin: 0; color: #0f172a;">{fac.get('facility_name', 'Unknown Facility')}</h4>
                        <div style="color: #3b82f6; font-size: 0.85rem; font-weight: bold; margin-bottom: 0.5rem;">{fac.get('type', 'Healthcare Facility')}</div>
                        <p style="margin: 0.2rem 0; color: #475569;">📍 {fac.get('address', 'N/A')}</p>
                        <p style="margin: 0.2rem 0; color: #475569;">📞 {fac.get('phone', 'N/A')}</p>
                        <p style="margin: 0.2rem 0; color: #475569;">🗣️ Languages: {', '.join(fac.get('languages_supported', []))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except FileNotFoundError:
            st.error("Facilities database (partner_facilities.json) could not be found.")

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True) # End main-container
