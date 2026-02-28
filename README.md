# MEDINOTED AI

A professional, dual-feature web application that functions as both a Clinical Note Cleaner and an AI Health Diary Summarizer.

## Features
### AI Health Diary Summarizer
- **Voice Recording:** Directly record your dictation from the browser interface.
- **High-Quality Transcription:** Uses OpenAI's Whisper model for accurate speech-to-text conversion.
- **SOAP Formatting:** Automatically categorizes your transcribed text.
- **Daily Entries:** Log how you're feeling and your daily activities.
- **Smart Analysis:** Automatically calculates a sentiment score (using `vaderSentiment`) to track your mood and auto-tags entries with relevant keywords (symptoms, mood, food).
- **Data Visualization:** View a longitudinal graph of your sentiment scores over time using `matplotlib`.
- **AI Suggestions:** Get intelligent, rule-based feedback (e.g., warnings to see a clinician if symptoms are persistently recorded) based on your history.

## Prerequisites
- Python 3.8 or higher
- An OpenAI API Key (required for the SOAP Note feature)

## Installation Instructions

1. **Navigate to the Project Directory:**
   ```bash
   cd "Clinical note cleaner"
   ```

2. **(Optional) Create a Virtual Environment:**
   It is recommended to run the app in a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This includes `streamlit`, `openai`, `vaderSentiment`, and `matplotlib`.*

## Running the Application

1. **Start the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your Browser:**
   The application will automatically open in your default web browser (typically at `http://localhost:8501`).

3. **Usage:**
   - On the left sidebar, enter your **OpenAI API Key** (if using the SOAP functionality).
   - Use the **tabs** at the top of the interface to switch between the Clinical Note Cleaner and the AI Health Diary.
   - For the Diary, enter a new entry and click "Save Entry". The app will store your data locally in `diary_entries.json` and immediately update your dashboard.


Screenshot
<img width="2914" height="1624" alt="image" src="https://github.com/user-attachments/assets/a8c89737-119c-4632-bde5-f11c5887f66f" />

<img width="2940" height="1642" alt="image" src="https://github.com/user-attachments/assets/afa693aa-abe2-45dc-bc5d-2296ea5f434b" />

<img width="2930" height="1562" alt="image" src="https://github.com/user-attachments/assets/2f80688b-7865-4f45-8649-7bfb04746afd" />


OUR deployment:
www.medinoted.com
