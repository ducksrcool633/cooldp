# ==============================
# Streamlit App: Parkinson's Spiral Prediction from Image
# ==============================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

st.title("Parkinson's Spiral Hand Prediction")
st.write("Upload a spiral drawing image, and the app will predict Parkinson's probability.")

# ------------------------------
# Step 1: Load trained model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("handpd_model.pkl")  # Upload your trained model in the same folder

clf = load_model()

# ------------------------------
# Step 2: Upload Image
# ------------------------------
uploaded_file = st.file_uploader("Upload a spiral image (PNG/JPG)", type=["png", "jpg", "jpeg"])

def extract_features(image):
    """
    Simple example feature extraction:
    - RMS: overall intensity variation
    - MAX_HT / MIN_HT / STD_HT: vertical stroke measures
    - MRT: rough measure of stroke smoothness
    You can expand this with more advanced features from OpenCV
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Flatten pixels for RMS
    pixels = gray.flatten()
    RMS = np.sqrt(np.mean(np.square(pixels)))
    
    MAX_HT = np.max(pixels)
    MIN_HT = np.min(pixels)
    STD_HT = np.std(pixels)
    
    # Simplified MRT = mean abs diff of adjacent pixels
    MRT = np.mean(np.abs(np.diff(pixels)))
    
    # Placeholder features for other model inputs (you can replace with better feature extraction)
    MAX_BETWEEN_ET_HT = RMS / 2
    MIN_BETWEEN_ET_HT = RMS / 4
    STD_DEVIATION_ET_HT = STD_HT / 2
    CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT = int(np.sum(pixels > 127))  # rough
    
    features = np.array([RMS, MAX_BETWEEN_ET_HT, MIN_BETWEEN_ET_HT, STD_DEVIATION_ET_HT,
                         MRT, MAX_HT, MIN_HT, STD_HT, CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT]).reshape(1, -1)
    
    return features

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Spiral", use_column_width=True)
    
    # ------------------------------
    # Step 3: Feature Extraction
    # ------------------------------
    features = extract_features(image)
    
    # ------------------------------
    # Step 4: Predict
    # ------------------------------
    prediction = clf.predict(features)[0]
    probability = clf.predict_proba(features)[0]
    
    class_map = {0: "Healthy", 1: "Parkinson's"}
    
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {class_map[prediction]}")
    st.write(f"**Probability:** Healthy = {probability[0]:.2f}, Parkinson's = {probability[1]:.2f}")
else:
    st.info("Upload a spiral drawing image to get prediction.")
