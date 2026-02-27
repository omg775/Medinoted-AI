# Deployment Guide - Health Assistant AI

This guide explains how to deploy your "Health Assistant AI" application to **Streamlit Community Cloud**.

## 1. Prepare for Deployment

### GitHub Repository
1. **Create a private GitHub repository.**
2. **Push your code** (excluding the `venv` or `.venv` folders and `.json` data files).
   - Ensure `requirements.txt` is in the root directory.
   - Ensure `app.py` is in the root directory.

### .gitignore (Recommended)
Create a `.gitignore` file to avoid uploading sensitive or unnecessary files:
```text
.venv/
venv/
__pycache__/
*.json
.env
```

## 2. Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Connect your GitHub account.
3. Click "New app" and select your repository, branch, and `app.py` as the main file.
4. **IMPORTANT: Set up Secrets.**
   - Before clicking "Deploy", go to **Advanced settings** -> **Secrets**.
   - Add your OpenAI API Key as follows:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
5. Click **Deploy**.

## 3. Data Persistence Warning

> [!WARNING]
> This app currently uses local JSON files (`notes.json`, `diary_entries.json`) to store data.
> **Streamlit Community Cloud uses ephemeral storage.** This means all saved notes will be DELETED every time the app server restarts (which happens periodically).

### How to solve this?
To save data permanently, you should connect to a cloud database such as:
- **Supabase** (PostgreSQL)
- **Google Sheets** (easiest for small apps)
- **Firebase/Firestore**

I can help you implement Google Sheets or Supabase integration if you want to keep your data permanently in the future!

## 4. Local Testing
To test deployment readiness locally:
```bash
.\.venv\Scripts\streamlit run app.py
```
Check that the "Configuration" section in the sidebar shows "API Key loaded from Secrets" (if you have a `.streamlit/secrets.toml` file locally).
