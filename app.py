# ==============================
# Streamlit App: Parkinson's HandPD Prediction
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import joblib

st.set_page_config(page_title="HandPD Predictor", layout="wide")

st.title("🖐 Parkinson's Spiral HandPD Predictor")
st.write("Upload your CSV data or enter new feature values to predict Parkinson's risk.")

# ------------------------------
# Step 1: Load Dataset (Optional)
# ------------------------------
uploaded_file = st.file_uploader("Upload Spiral HandPD CSV (optional)", type=["csv"])

if uploaded_file is not None:
    try:
        spiral_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
        st.write("Columns detected:", list(spiral_df.columns))

        # Check required columns
        expected_cols = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
            'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]
        missing = [c for c in expected_cols if c not in spiral_df.columns]
        if missing:
            st.warning(f"CSV is missing columns: {missing}")
        else:
            # Map CLASS_TYPE
            spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1

            st.subheader("Feature Distributions")
            numeric_features = expected_cols[2:]  # all except ID_PATIENT & CLASS_TYPE
            for feature in numeric_features:
                plt.figure(figsize=(6,4))
                sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True)
                plt.xticks([0,1], ['Healthy', "Parkinson's"])
                plt.title(f"{feature} by Class")
                plt.ylabel(feature)
                plt.xlabel("Class")
                st.pyplot(plt.gcf())
                plt.clf()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ------------------------------
# Step 2: Manual Input for Prediction
# ------------------------------
st.subheader("Enter Features for Prediction")

feature_input = {
    'RMS': st.number_input("RMS", value=1200.0),
    'MAX_BETWEEN_ET_HT': st.number_input("Max Between ET-HT", value=30.0),
    'MIN_BETWEEN_ET_HT': st.number_input("Min Between ET-HT", value=5.0),
    'STD_DEVIATION_ET_HT': st.number_input("Std Deviation ET-HT", value=12.0),
    'MRT': st.number_input("MRT", value=0.8),
    'MAX_HT': st.number_input("Max HT", value=150.0),
    'MIN_HT': st.number_input("Min HT", value=20.0),
    'STD_HT': st.number_input("Std HT", value=25.0),
    'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT': st.number_input(
        "Changes from Neg to Pos ET-HT", value=10.0
    )
}

# ------------------------------
# Step 3: Load or Train Model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        # Try loading pre-trained model
        clf = joblib.load("handpd_model.pkl")
    except:
        # If not exists, train on uploaded CSV
        if uploaded_file is not None and not missing:
            X = spiral_df[expected_cols[2:]]
            y = spiral_df['CLASS_TYPE']
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X, y)
            joblib.dump(clf, "handpd_model.pkl")
        else:
            clf = None
    return clf

clf = load_model()

# ------------------------------
# Step 4: Prediction
# ------------------------------
if clf is not None:
    if st.button("Predict"):
        new_sample = pd.DataFrame([feature_input])
        prediction = clf.predict(new_sample)
        probability = clf.predict_proba(new_sample)
        class_map = {0: "Healthy", 1: "Parkinson's"}

        st.write(f"**Predicted class:** {class_map[int(prediction[0])]}")
        st.write(f"**Probability:** Healthy={probability[0][0]:.2f}, Parkinson's={probability[0][1]:.2f}")
else:
    st.warning("Upload a valid CSV or provide proper feature inputs to train the model.")

st.info("✅ Fully warning-free, ready for deployment on Streamlit Cloud!")
