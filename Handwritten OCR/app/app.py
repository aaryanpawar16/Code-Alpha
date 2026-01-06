import os
import cv2
import numpy as np
import pickle
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tf_keras.models import load_model

# --- CRITICAL FIX ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# --------------------

# --- Configuration ---
MODEL_PATH = "models/character_model.h5"
LABEL_MAP_PATH = "data/processed/label_map.pkl"

st.set_page_config(page_title="Handwritten OCR", page_icon="✍️", layout="wide")

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = load_model(MODEL_PATH, compile=False)
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_map = pickle.load(f)
    return model, label_map

def preprocess_canvas(image_data, rotate=False, flip=False):
    """
    Input: RGBA image from the canvas.
    Output: (1, 28, 28, 1) Normalized Grayscale for the model.
    """
    # 1. Convert to Grayscale
    img = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    
    # 2. Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # 3. Invert (Black text on White bg -> White text on Black bg)
    img = cv2.bitwise_not(img)
    
    # 4. Normalize
    img = img.astype('float32') / 255.0
    
    # 5. Apply User Adjustments (To match EMNIST format)
    if rotate:
        img = np.transpose(img) # This usually fixes EMNIST
    if flip:
        img = np.flipud(img)    # Flip upside down if needed
    
    # 6. Reshape for Model
    model_input = img.reshape(1, 28, 28, 1)
    
    # 7. Create a nice debug view for the user
    debug_view = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
    return model_input, debug_view

# --- UI Layout ---
st.title("✍️ Handwritten Character Recognition")

# Load Model
model, label_map = load_resources()

if model is None:
    st.error("Model not found! Please run `python src/train.py` first.")
else:
    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Settings")
    stroke_width = st.sidebar.slider("Stroke Width", 10, 30, 20)
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Orientation Fix")
    st.sidebar.write("EMNIST images are often rotated. Tweak these if predictions are wrong:")
    do_rotate = st.sidebar.checkbox("Transpose (Rotate)", value=True)
    do_flip = st.sidebar.checkbox("Flip Vertical", value=False)
    
    # --- Main Area ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### 1. Draw Here:")
        canvas_result = st_canvas(
            fill_color="#ffffff",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#ffffff",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        st.write("### 2. Model 'See' View:")
        # Placeholder for the debug image
        debug_placeholder = st.empty()
        
        if canvas_result.image_data is not None and np.max(canvas_result.image_data) > 0:
            # Preprocess live
            processed_img, debug_view = preprocess_canvas(
                canvas_result.image_data, 
                rotate=do_rotate, 
                flip=do_flip
            )
            
            # Show what the model sees
            debug_placeholder.image(debug_view, caption="Processed Input (28x28)", clamp=True)
            
            # Predict Button
            if st.button("Predict Character", type="primary"):
                prediction = model.predict(processed_img, verbose=0)
                predicted_idx = np.argmax(prediction)
                confidence = np.max(prediction)
                char = label_map[predicted_idx]
                
                # Result
                st.success(f"## Prediction: **{char}**")
                st.write(f"Confidence: **{confidence*100:.2f}%**")
                
                # Debugging Hint
                if confidence < 0.60:
                    st.warning("⚠️ Low confidence? Try toggling 'Transpose' or 'Flip' in the sidebar!")