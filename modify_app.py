import re

with open("app.py", "r") as f:
    code = f.read()

# 1) Add imports
imports_to_add = """import bcrypt
"""
code = code.replace("import streamlit as st", imports_to_add + "import streamlit as st")

# 2) Replace file storage & state initialization
storage_section_regex = r'# -----------------------------------------------------------------------------\n# File Storage Functions.*?# -----------------------------------------------------------------------------'

new_storage_section = """# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------"""
code = re.sub(storage_section_regex, new_storage_section, code, flags=re.DOTALL)

# Add session state variables
state_init_regex = r'if "last_processed_audio" not in st\.session_state:\n    st\.session_state\["last_processed_audio"\] = None'
new_state_init = """if "last_processed_audio" not in st.session_state:
    st.session_state["last_processed_audio"] = None
if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"
"""
code = code.replace('if "last_processed_audio" not in st.session_state:\n    st.session_state["last_processed_audio"] = None', new_state_init)

# Remove NOTES_FILE = "notes.json"
code = code.replace('NOTES_FILE = "notes.json"\n', '')

# Replace "Main UI Construction" to include auth check
auth_ui = """# -----------------------------------------------------------------------------
# Authentication UI
# -----------------------------------------------------------------------------
if not st.session_state["is_authenticated"]:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown(""" + '"""<div class="header-wrapper"><h1 class="main-title">Health Assistant AI</h1></div>"""' + """, unsafe_allow_html=True)
    
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
"""
code = code.replace("""# -----------------------------------------------------------------------------
# Main UI Construction
# -----------------------------------------------------------------------------""", auth_ui)

# Add logout button in sidebar
sidebar_logout_ui = """with st.sidebar:
    st.markdown(f"**Logged in as: {st.session_state['username']}**")
    if st.button("Logout"):
        st.session_state["is_authenticated"] = False
        st.session_state["username"] = None
        st.session_state["chat_history"] = []
        st.session_state["transcribed_text"] = ""
        st.rerun()
    st.divider()"""
    
code = code.replace("with st.sidebar:", sidebar_logout_ui)

# Remove the file
with open("app_new.py", "w") as f:
    f.write(code)

