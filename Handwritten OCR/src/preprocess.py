import os
import numpy as np
import pickle
from emnist import extract_training_samples, extract_test_samples

# --- ENVIRONMENT FIX ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# -----------------------

# NOW import keras (it will use the legacy version)
from tensorflow.keras.utils import to_categorical

# --- Config ---
OUTPUT_PATH = "data/processed/"

LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    print("Downloading/Loading EMNIST (Balanced) dataset...")
    X_train_raw, y_train_raw = extract_training_samples('balanced')
    X_test_raw, y_test_raw = extract_test_samples('balanced')

    # Normalize
    X_train = X_train_raw.astype('float32') / 255.0
    X_test = X_test_raw.astype('float32') / 255.0

    # Reshape
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # One-Hot Encode
    num_classes = len(LABEL_MAP)
    y_train = to_categorical(y_train_raw, num_classes)
    y_test = to_categorical(y_test_raw, num_classes)

    # Save
    print("Saving processed data...")
    np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)
    
    with open(os.path.join(OUTPUT_PATH, 'label_map.pkl'), 'wb') as f:
        pickle.dump(LABEL_MAP, f)

    print(f"Success! Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()