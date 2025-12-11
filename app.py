# app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title("Parkinson's Spiral Prediction App")
st.write("""
This app predicts the probability of Parkinson's from spiral hand drawings or numeric features.
You can either upload a **spiral image** or a **CSV file with precomputed features**.
""")

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("handpd_model.pkl")  # make sure this file is in your repo

clf = load_model()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.header("Predict from Spiral Image")
uploaded_image = st.file_uploader("Upload a spiral image", type=["png","jpg","jpeg"], key="image")

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("L")  # convert to grayscale
    st.image(img, caption="Uploaded Spiral", use_column_width=True)

    # Resize / flatten image to match model input (adjust if your model expects different size/features)
    img_resized = img.resize((28,28))  # example size
    img_array = np.array(img_resized).flatten().reshape(1, -1)

    try:
        pred_prob = clf.predict_proba(img_array)[0]
        class_map = {0:"Healthy", 1:"Parkinson's"}
        pred_class = class_map[np.argmax(pred_prob)]
        st.write(f"**Predicted class:** {pred_class}")
        st.write(f"**Probability:** Healthy={pred_prob[0]:.2f}, Parkinson's={pred_prob[1]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# CSV FEATURE UPLOAD
# -------------------------------
st.header("Predict from CSV Features")
uploaded_csv = st.file_uploader("Upload CSV with features", type=["csv"], key="csv")

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.write("CSV Preview:")
        st.dataframe(df.head())

        # Required features
        expected_features = [
            'RMS', 
            'MAX_BETWEEN_ET_HT', 
            'MIN_BETWEEN_ET_HT',
            'STD_DEVIATION_ET_HT', 
            'MRT', 
            'MAX_HT', 
            'MIN_HT', 
            'STD_HT', 
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]

        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        else:
            X = df[expected_features].values
            pred_prob = clf.predict_proba(X)
            class_map = {0:"Healthy", 1:"Parkinson's"}

            results = []
            for i, prob in enumerate(pred_prob):
                pred_class = class_map[np.argmax(prob)]
                results.append({
                    "Predicted Class": pred_class,
                    "Healthy Probability": round(prob[0],2),
                    "Parkinson's Probability": round(prob[1],2)
                })
            st.write(pd.DataFrame(results))
    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")
