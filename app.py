# app.py
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------
# Title
# -------------------------
st.title("Parkinson's Spiral Hand Drawing Predictor")

st.write("""
Upload an image of a spiral you drew. The app will predict the probability
of Parkinson's-related tremor based on the drawing.
""")

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Upload Spiral Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Open and display the image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to fixed size for feature extraction
    image = image.resize((100, 100))
    img_array = np.array(image)
    
    # -------------------------
    # Extract simple features
    # -------------------------
    # These are just demo features; in a real model, you'd use more sophisticated ones
    RMS = np.sqrt(np.mean(img_array**2))
    MAX_VAL = np.max(img_array)
    MIN_VAL = np.min(img_array)
    STD_VAL = np.std(img_array)
    MEAN_VAL = np.mean(img_array)

    features = np.array([[RMS, MAX_VAL, MIN_VAL, STD_VAL, MEAN_VAL]])

    st.write("### Extracted Features:")
    st.write({
        "RMS": round(RMS, 2),
        "Max Pixel": int(MAX_VAL),
        "Min Pixel": int(MIN_VAL),
        "Std Dev": round(STD_VAL, 2),
        "Mean Pixel": round(MEAN_VAL, 2)
    })

    # -------------------------
    # Dummy Model (for demo)
    # -------------------------
    # Normally you'd load a pre-trained model
    # For demo purposes, we train a small Random Forest with random data
    X_dummy = np.random.rand(50, 5) * 255
    y_dummy = np.random.randint(0, 2, 50)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_dummy, y_dummy)

    # Predict probability
    prob = clf.predict_proba(features)[0]
    st.write("### Prediction Probabilities:")
    st.write({
        "Healthy": f"{prob[0]*100:.2f}%",
        "Parkinson's": f"{prob[1]*100:.2f}%"
    })

    # Show the predicted class
    predicted_class = "Healthy" if prob[0] > prob[1] else "Parkinson's"
    st.success(f"Predicted Class: {predicted_class}")
