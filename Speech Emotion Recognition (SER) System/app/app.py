import os
import sys

# --- CRITICAL FIX For Windows/Python 3.12 ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# --------------------------------------------

import streamlit as st
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_PATH = "models/emotion_model.h5"
LABEL_ENCODER_PATH = "data/processed/label_encoder.pkl"

# --- Helper Functions ---
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    # We compile=False because we only need prediction, not training
    model = load_model(MODEL_PATH, compile=False)
    return model

@st.cache_resource
def load_label_encoder():
    if not os.path.exists(LABEL_ENCODER_PATH):
        st.error(f"Label Encoder not found at {LABEL_ENCODER_PATH}")
        return None
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        lb = pickle.load(f)
    return lb

def extract_features(audio_file):
    # Librosa load expects a filename or a file-like object
    # Force 22050Hz to match training
    audio, sample_rate = librosa.load(audio_file, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Reshape for CNN: (1, 40, 1)
    mfccs_cnn = np.expand_dims(mfccs_scaled, axis=0)
    mfccs_cnn = np.expand_dims(mfccs_cnn, axis=2)
    return mfccs_cnn

# --- UI Layout ---
st.title("🎤 Speech Emotion Recognition")
st.write("Upload an audio file (.wav) to detect the speaker's emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    # 1. Display Audio Player
    st.audio(uploaded_file, format='audio/wav')

    # 2. Load Resources
    model = load_emotion_model()
    lb = load_label_encoder()

    if model is not None and lb is not None:
        # 3. Analyze Button
        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing audio patterns..."):
                try:
                    # Preprocess
                    features = extract_features(uploaded_file)
                    
                    # Predict
                    prediction = model.predict(features, verbose=0)
                    predicted_index = np.argmax(prediction)
                    predicted_label = lb.inverse_transform([predicted_index])[0]
                    confidence = prediction[0][predicted_index]

                    # 4. Display Results
                    st.success(f"**Predicted Emotion:** {predicted_label.upper()}")
                    st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")
                    
                    # Probability Distribution Chart
                    st.write("---")
                    st.write("Probability Distribution:")
                    probs = prediction[0]
                    classes = lb.classes_
                    st.bar_chart(dict(zip(classes, probs)))

                except Exception as e:
                    st.error(f"Error processing audio: {e}")