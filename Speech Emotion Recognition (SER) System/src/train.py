import os

# --- CRITICAL FIX For Windows/Python 3.12 ---
# This forces TensorFlow to use the stable "Legacy" Keras
# which does not have the Overflow/Int error.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# --------------------------------------------

import numpy as np
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# Note: We must import the model builder AFTER setting the environment variable
from model_builder import create_model

# Paths
DATA_PATH = "data/processed/"
MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load Data
    print("Loading preprocessed data...")
    if not os.path.exists(os.path.join(DATA_PATH, 'X_train.npy')):
        print("Error: processed data not found. Run src/preprocess.py first.")
        return

    X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

    # 2. Reshape for CNN
    # The CNN expects 3D input: (Batch_Size, Steps, Channels)
    x_train_cnn = np.expand_dims(X_train, axis=2)
    x_test_cnn = np.expand_dims(X_test, axis=2)

    # 3. Build Model
    num_classes = y_train.shape[1]
    input_shape = (x_train_cnn.shape[1], 1)
    
    model = create_model(input_shape, num_classes)
    model.summary()

    # 4. Define Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=1, patience=2, min_lr=0.0000001)

    # 5. Train
    print("Starting training...")
    history = model.fit(x_train_cnn, y_train, 
                        batch_size=64, 
                        epochs=50, 
                        validation_data=(x_test_cnn, y_test), 
                        callbacks=[checkpoint, rlrp])

    # 6. Save History
    with open(os.path.join(MODEL_DIR, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"Training complete. Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()