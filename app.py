# ==============================
# Streamlit App: Parkinson's Spiral Hand Prediction
# ==============================

import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image

# ==============================
# Page Setup
# ==============================
st.set_page_config(
    page_title="Parkinson's Spiral Predictor",
    layout="centered"
)

st.title("🖊 Parkinson's Spiral Hand Predictor")
st.write("Upload an image of a spiral drawing, and the model will predict the probability of Parkinson's.")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model_path = "handpd_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found! Make sure it is in your repo.")
        return None
    return joblib.load(model_path)

clf = load_model()
if clf is None:
    st.stop()

# ==============================
# Image Upload
# ==============================
uploaded_file = st.file_uploader("Upload a spiral image (jpg, png)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file).convert("L")  # convert to grayscale
        st.image(image, caption="Uploaded Spiral", use_column_width=True)

        # ==============================
        # Preprocess Image
        # ==============================
        # Example preprocessing: flatten image and normalize
        img_array = np.array(image.resize((128,128)))  # resize to 128x128
        img_array = img_array.flatten() / 255.0        # normalize to 0-1
        img_array = img_array.reshape(1, -1)          # shape (1, features)

        # ==============================
        # Predict
        # ==============================
        prediction = clf.predict(img_array)[0]
        probability = clf.predict_proba(img_array)[0]

        class_map = {0: "Healthy", 1: "Parkinson's"}

        st.write(f"**Predicted class:** {class_map[prediction]}")
        st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

else:
    st.info("Please upload a spiral image to get a prediction.")
