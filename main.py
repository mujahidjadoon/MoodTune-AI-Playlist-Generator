import streamlit as st
import requests
import time
import torch
import numpy as np
import random
import os
import threading
from flask import Flask, request, jsonify

# --- 1. CONFIGURATION AND MODEL SETUP ---

# Flask app configuration
API_HOST = '127.0.0.1'
API_PORT = 5000
API_URL = f"http://{API_HOST}:{API_PORT}/detect_mood"
FLASK_APP = Flask(__name__)

# Mood categories
MOOD_CATEGORIES = ['Happy', 'Sad', 'Energetic', 'Calm', 'Romantic']

# --- 2. PLACEHOLDER MODEL LOADING ---

# Use Hugging Face/Transformers (compatible with PyTorch) for text analysis
# NOTE: You MUST have the 'transformers' library installed for this to work.
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    print(f"✅ Model Loaded and running on: {DEVICE}")
except Exception as e:
    print(f"⚠️ Could not load Transformer model. Using a DUMMY fallback. Error: {e}")
    model = None


# --- 3. PYTORCH MOOD CLASSIFICATION FUNCTION ---
def classify_mood_pytorch(text):
    """Placeholder for your PyTorch Text Classification Logic."""
    if not text:
        return None

    if model:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

        # Mapping (Placeholder logic)
        if predicted_class_id == 1:  # Positive sentiment
            return random.choice(['Happy', 'Energetic', 'Romantic'])
        else:  # Negative sentiment
            return random.choice(['Sad', 'Calm'])
    else:
        # DUMMY fallback if model failed to load
        return random.choice(MOOD_CATEGORIES)


# --- 4. FLASK API ENDPOINT ---
@FLASK_APP.route('/detect_mood', methods=['POST'])
def detect_mood():
    """API endpoint to receive text and return the detected mood."""
    data = request.get_json()
    text_input = data.get('text', '')

    if not text_input:
        return jsonify({"mood": None, "error": "No text provided"}), 400

    detected_mood = classify_mood_pytorch(text_input)

    return jsonify({"mood": detected_mood})


# --- 5. BACKGROUND THREAD FOR FLASK API ---

# Function to run Flask server in a separate thread
def run_flask_api():
    # We MUST use use_reloader=False to run Flask inside another Python process (Streamlit)
    FLASK_APP.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False)


# Start the Flask API server in the background thread if not already running
if 'api_thread' not in st.session_state:
    print("Starting Flask API in background thread...")
    api_thread = threading.Thread(target=run_flask_api)
    api_thread.daemon = True
    api_thread.start()
    st.session_state['api_thread'] = api_thread
    time.sleep(1)  # Give the server a moment to start


# --- 6. STREAMLIT UI ---

# Dummy playlist suggestions
def get_playlist(mood):
    if mood == 'Happy': return ["Sunroof - Nicky Youre", "Good as Hell - Lizzo", "Walking on Sunshine"]
    if mood == 'Sad': return ["Hallelujah - Leonard Cohen", "Fix You - Coldplay", "Someone Like You - Adele"]
    if mood == 'Energetic': return ["Blinding Lights - The Weeknd", "Seven Nation Army - The White Stripes",
                                    "Uptown Funk - Bruno Mars"]
    if mood == 'Calm': return ["Weightless - Marconi Union", "Clair de Lune - Debussy", "Enya - Orinoco Flow"]
    if mood == 'Romantic': return ["Perfect - Ed Sheeran", "At Last - Etta James", "Thinking Out Loud - Ed Sheeran"]
    return []


# Mood data for visualization
MOOD_DATA = {
    'Happy': {'emoji': '😊', 'background': 'yellow'},
    'Sad': {'emoji': '😢', 'background': 'blue'},
    'Energetic': {'emoji': '⚡', 'background': 'red'},
    'Calm': {'emoji': '🧘', 'background': 'green'},
    'Romantic': {'emoji': '💖', 'background': 'pink'},
}

# --- Streamlit Front-End Interface ---

st.set_page_config(page_title="MoodTune AI", layout="centered")
st.title("🎵 MoodTune AI Playlist Generator")
st.caption(f"AI powered by PyTorch/Transformers running locally via API at port {API_PORT}")
st.markdown("---")

text_input = st.text_area(
    "What's on your mind? (Type a sentence or a paragraph)",
    height=150,
    placeholder="Example: 'I'm excited to start a new day!'"
)

if st.button("✨ Generate Playlist"):
    if not text_input:
        st.error("Please enter some text to detect your mood.")
    else:
        with st.spinner('Analyzing Mood with PyTorch Model...'):
            try:
                # 1. Send text to the local API
                response = requests.post(API_URL, json={'text': text_input})

                if response.status_code == 200:
                    result = response.json()
                    current_mood = result.get('mood')

                    if current_mood and current_mood in MOOD_DATA:
                        data = MOOD_DATA[current_mood]

                        st.success(f"Mood Detected: {data['emoji']} **{current_mood}**")
                        st.markdown(f"### Playlist for your {current_mood} Vibe:")

                        playlist = get_playlist(current_mood)
                        st.code('\n'.join(playlist), language='text')

                    else:
                        st.warning("Mood was detected, but could not be mapped to a playlist category.")

                elif response.status_code == 404:
                    st.error("API Error 404: Endpoint not found. Check the Flask API code.")

                else:
                    st.error(f"API Error ({response.status_code}): Could not process the request.")

            except requests.exceptions.ConnectionError:
                st.error(f"🚨 Connection Failed: Flask server failed to start or is unresponsive on port {API_PORT}.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
