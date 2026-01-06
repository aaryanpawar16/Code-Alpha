import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = "models/emotion_model.h5"
LABEL_ENCODER_PATH = "data/processed/label_encoder.pkl"

def predict_emotion(audio_path):
    """
    Predicts the emotion of a single audio file.
    """
    # 1. Load Model and Label Encoder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return "Error: Model or Label Encoder not found. Run train.py first.", 0.0

    model = load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        lb = pickle.load(f)

    # 2. Extract Features (must match training logic)
    try:
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        return f"Error reading audio file: {e}", 0.0
    
    # 3. Reshape for CNN
    mfccs_cnn = np.expand_dims(mfccs_scaled, axis=0) # Add batch dimension
    mfccs_cnn = np.expand_dims(mfccs_cnn, axis=2)    # Add channel dimension

    # 4. Predict
    prediction = model.predict(mfccs_cnn, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = lb.inverse_transform([predicted_index])[0]
    confidence = prediction[0][predicted_index]
    
    return predicted_label, confidence

if __name__ == "__main__":
    # Test with a dummy file path - replace this with a real path from your drive
    # For example, grab a file from the TESS dataset to see how it generalizes
    test_file = "C:/path/to/some/test_audio.wav" 
    
    if os.path.exists(test_file):
        emotion, conf = predict_emotion(test_file)
        print(f"\nPredicted Emotion: {emotion.upper()}")
        print(f"Confidence: {conf * 100:.2f}%")
    else:
        print("Please edit the 'test_file' variable in predict.py to point to a valid .wav file.")