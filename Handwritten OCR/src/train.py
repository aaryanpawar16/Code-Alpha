import os
import numpy as np
import pickle

# --- ENVIRONMENT FIX ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# -----------------------

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import create_model

DATA_PATH = "data/processed/"
MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "character_model.h5")

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading data...")
    if not os.path.exists(os.path.join(DATA_PATH, 'X_train.npy')):
        print("Error: processed data not found.")
        return

    X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

    input_shape = X_train.shape[1:] 
    num_classes = y_train.shape[1] 
    
    print(f"Building model for {num_classes} classes...")
    model = create_model(input_shape, num_classes)
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, rlrp, early_stop]
    )

    print(f"Training complete. Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()