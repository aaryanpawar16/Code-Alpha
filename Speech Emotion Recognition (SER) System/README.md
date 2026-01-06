# 🎤 Speech Emotion Recognition (SER) System

A Deep Learning application that detects human emotions (Happy, Sad, Angry, Fearful, etc.) from speech audio using **MFCC features** and a **1D Convolutional Neural Network (CNN)**.

This project uses the **RAVDESS** dataset and provides a web-based interface built with **Streamlit**.

---

## 🚀 Features
* **Emotion Detection:** Classifies audio into 8 distinct emotions: *Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised*.
* **Deep Learning Model:** Uses a robust 1D CNN architecture optimized for audio feature extraction.
* **Interactive UI:** A web-based interface to upload `.wav` files and view predictions instantly.
* **Confidence Scoring:** Displays the probability percentage for the predicted emotion.

---

## 📂 Project Structure
```text
Speech_Emotion_Recognition/
│
├── data/
│   └── processed/          # Stores extracted features (X_train.npy, label_encoder.pkl)
│
├── models/
│   └── emotion_model.h5    # The trained CNN model weights
│
├── src/
│   ├── preprocess.py       # Downloads data & extracts MFCC features
│   ├── model_builder.py    # Defines the CNN architecture
│   └── train.py            # Trains the model and saves it
│
├── app/
│   └── app.py              # Streamlit Web Interface
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🛠️ Installation & Setup
1. Clone the Repository
(If you downloaded this code manually, just open the terminal in the project folder).

Bash

cd Speech_Emotion_Recognition
2. Install Dependencies
Make sure you have Python 3.10 or newer installed.

Bash

pip install -r requirements.txt
(Windows Users: If you encounter errors with Keras/TensorFlow, ensure tf_keras is installed: pip install tf_keras)

🏃‍♂️ Usage Guide
Step 1: Download & Preprocess Data
This script automatically downloads the RAVDESS dataset from KaggleHub and processes the audio files into numerical data (MFCCs).

Bash

python src/preprocess.py
Input: Downloads ~400MB of audio data.

Output: Saves .npy files to data/processed/.

Step 2: Train the Model
Train the CNN model on the processed data.

Bash

python src/train.py
Output: Saves the trained model to models/emotion_model.h5.

Performance: Typically achieves ~95%+ accuracy on the training set.

Step 3: Run the Web App
Launch the user interface to test the model with real audio files.

Bash

streamlit run app/app.py
This will open http://localhost:8501 in your browser.

📊 Model Architecture
The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) (40 features per time step) from the audio.

CNN Layers:

Conv1D (256 filters) + Batch Normalization + Max Pooling

Conv1D (256 filters) + Batch Normalization + Max Pooling

Conv1D (128 filters) + Batch Normalization + Max Pooling

Flatten → Dense (32 units) → Output (8 units, Softmax)

🧪 How to Test
You can use the audio files downloaded in Step 1 to test the app. Navigate to: C:/Users/<YourUser>/.cache/kagglehub/datasets/.../audio_speech_actors_01-24/

Filename Code Guide (3rd Number):

03-01-01... → Neutral

03-01-03... → Happy

03-01-04... → Sad

03-01-05... → Angry

📜 License
This project is open-source and available for educational purposes.