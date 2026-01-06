import os
# --- CRITICAL FIX ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# --------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization

def create_model(input_shape, num_classes):
    """
    Builds a robust 1D CNN model for Emotion Recognition.
    """
    model = Sequential()

    # --- 1st Convolutional Block ---
    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    # --- 2nd Convolutional Block ---
    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    # --- 3rd Convolutional Block ---
    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    model.add(Dropout(0.2))

    # --- Classification Block ---
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model