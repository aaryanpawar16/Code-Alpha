import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# --- Configuration ---
# UPDATED PATH for User 'LENOVO'
DATASET_PATH = "C:/Users/LENOVO/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1"
OUTPUT_PATH = "data/processed/"

# We will use actors 23 and 24 exclusively for testing to ensure the model
# isn't just memorizing specific voices (Subject Independent Split).
TEST_ACTORS = [23, 24] 

def extract_mfcc(file_path):
    """
    Loads an audio file and extracts 40 MFCC features.
    """
    try:
        # Load audio using default backend (soxr) which is faster and doesn't need resampy
        # We explicitly set sr=22050 to ensure consistency across all files
        audio, sample_rate = librosa.load(file_path, sr=22050)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Take the mean across time (axis 1) to get a 1D array of 40 values
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(dataset_path):
    features = []
    labels = []
    actors = []

    print(f"Scanning dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: The folder {dataset_path} does not exist.")
        print("Please check if the download finished successfully or if the path is correct.")
        return np.array([]), np.array([]), np.array([])

    # Walk through the directory structure
    file_count = 0
    valid_files = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_count += 1
                # RAVDESS filename convention: 03-01-06-01-02-01-12.wav
                # The 3rd part (06) is the emotion code.
                parts = file.split("-")
                
                # Safety check for filename format
                if len(parts) < 7: continue

                try:
                    emotion_code = int(parts[2])
                    actor_id = int(parts[6].split(".")[0])
                except ValueError:
                    continue
                
                # Dictionary to map codes to text
                emotions = {
                    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                }
                emotion_label = emotions.get(emotion_code)

                # Extract features
                file_path = os.path.join(root, file)
                
                # Print progress every 100 files to show it's working
                if file_count % 100 == 0:
                    print(f"Processing file {file_count}...")
                    
                mfcc = extract_mfcc(file_path)
                
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(emotion_label)
                    actors.append(actor_id)
                    valid_files += 1

    print(f"Scanned {file_count} files. Successfully processed {valid_files} files.")
    return np.array(features), np.array(labels), np.array(actors)

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # 1. Load raw data
    X, y, actors = load_data(DATASET_PATH)
    
    # CRITICAL CHECK: Stop if no data found
    if len(X) == 0:
        print("CRITICAL ERROR: No audio data was processed.")
        return

    # 2. Split into Train and Test based on Actors
    test_mask = np.isin(actors, TEST_ACTORS)
    
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train_raw, y_test_raw = y[~test_mask], y[test_mask]

    # 3. Encode the labels (text -> numbers -> one-hot vectors)
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train_raw))
    # Handle case where test set might miss some classes
    y_test = to_categorical(lb.transform(y_test_raw))

    # 4. Save the Label Encoder (vital for predicting later)
    with open(os.path.join(OUTPUT_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(lb, f)

    # 5. Save the processed arrays
    np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)

    print(f"\n--- SUCCESS ---")
    print(f"Data saved to {OUTPUT_PATH}")
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    print("You can now run 'python src/train.py'")

if __name__ == "__main__":
    main()