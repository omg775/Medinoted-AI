FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for PyAudio, pydub (ffmpeg), and others
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libasound2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip to ensure it finds pre-compiled wheels (avoids slow source compilations)
RUN pip install --no-cache-dir --upgrade pip

# Install the Python dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Install the a specific scispacy model used in app.py
RUN pip install --no-cache-dir https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8000

# Command to run the Streamlit app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
