import os
import cv2
import numpy as np
import pickle
# STANDARD IMPORT
from keras.models import load_model

MODEL_PATH = "models/character_model.h5"
LABEL_MAP_PATH = "data/processed/label_map.pkl"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first!")
        
    model = load_model(MODEL_PATH)
    
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
        
    return model, label_map

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_character(image_path):
    model, label_map = load_resources()
    processed_img = preprocess_image(image_path)
    
    prediction = model.predict(processed_img, verbose=0)
    predicted_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    
    char = label_map[predicted_idx]
    return char, confidence

if __name__ == "__main__":
    test_img = "test_sample.png" 
    if os.path.exists(test_img):
        char, conf = predict_character(test_img)
        print(f"Predicted: {char} ({conf*100:.2f}%)")
    else:
        print("Set 'test_img' to a valid path to test.")