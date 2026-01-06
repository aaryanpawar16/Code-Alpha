# ✍️ Handwritten Character Recognition (OCR)

An interactive Deep Learning application that recognizes handwritten digits (0-9) and letters (A-Z, a-z) drawn by the user in real-time.

Built with **TensorFlow/Keras**, **EMNIST Dataset**, and **Streamlit**.

---

## 🚀 Features
* **Real-time Recognition:** Draw on a digital canvas and get instant predictions.
* **47 Classes:** Supports Digits (0-9), Uppercase (A-Z), and Lowercase (a-z) characters.
* **Deep CNN:** Powered by a Convolutional Neural Network trained on the EMNIST Balanced dataset.
* **Visual Debugging:** Shows exactly what the model "sees" (preprocessed 28x28 grid).
* **Orientation Controls:** Sidebar settings to handle EMNIST's rotation quirks (Transpose/Flip).

---

## 📂 Project Structure
```text
Handwritten_OCR/
│
├── data/ (https://drive.google.com/drive/folders/14PL2uo20dwzI5_O3IJo8andV45ysASio?usp=sharing)
│   └── processed/          # Stores normalized EMNIST data & label maps
│
├── models/
│   └── character_model.h5  # The trained CNN model
│
├── src/
│   ├── preprocess.py       # Downloads & processes EMNIST data
│   ├── model.py            # CNN Architecture definition
│   ├── train.py            # Model training script
│   └── predict.py          # CLI prediction logic
│
├── app/
│   └── app.py              # Streamlit Web App with Drawing Canvas
│
└── README.md

🛠️ Installation
Prerequisites: Python 3.10 or newer.

1. Create a Virtual Environment (Highly Recommended) To avoid conflicts with TensorFlow versions on Windows, use a clean environment:

Bash

python -m venv .venv
.venv\Scripts\activate
2. Install Dependencies Install the specific versions that work with the legacy Keras support:

Bash

pip install tensorflow==2.16.1 tf_keras==2.16.0 numpy==1.26.4 opencv-python emnist scikit-learn streamlit streamlit-drawable-canvas
🏃‍♂️ Usage Guide
Step 1: Prepare Data
Downloads the EMNIST dataset and converts it to NumPy arrays. (Note: If the download hangs, manually place emnist.zip in your .cache/emnist folder).

Bash

python src/preprocess.py
Step 2: Train Model
Trains the CNN for 15-20 epochs.

Bash

python src/train.py
Performance: Targets ~88% Validation Accuracy.

Output: Saves models/character_model.h5.

Step 3: Run the App
Launches the web interface in your browser.

Bash

streamlit run app/app.py
⚙️ How to Fix Wrong Predictions (Rotation Issue)
The EMNIST dataset is stored with images rotated 90° and flipped. Sometimes, your drawing might be fed into the model sideways (e.g., an "L" becomes a "J", or "Z" becomes "N").

Solution:

Open the Sidebar (Arrow > on top-left).

Look at the "Model 'See' View" on the right side of the screen.

Adjust the Orientation Fix checkboxes:

Transpose (Rotate): Uncheck this if your letter looks sideways.

Flip Vertical: Check this if your letter looks upside down.

Tweak until the black-and-white image looks like the letter you drew!

🧠 Model Architecture
Input: 28x28 Grayscale Image

Layer 1: Conv2D (32 filters) + BatchNorm + MaxPool

Layer 2: Conv2D (64 filters) + BatchNorm + MaxPool + Dropout

Classifier: Flatten → Dense (128) → Output (Softmax 47 Classes)

Optimizer: SGD (Stochastic Gradient Descent) for stability.

📜 License

Open-source project for educational purposes.
