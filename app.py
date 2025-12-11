# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Parkinson's Hand Tremor Prediction")

# =========================
# Step 1: Upload CSV
# =========================
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Keep only the expected columns
        expected_cols = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT',
            'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT',
            'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]
        df = df[expected_cols]

        st.success("CSV loaded successfully!")
        st.write(df.head())

        # =========================
        # Step 2: Train model
        # =========================
        feature_cols = expected_cols[2:]  # all except ID_PATIENT and CLASS_TYPE
        X = df[feature_cols]
        y = df['CLASS_TYPE']

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)

        st.success("Model trained on your dataset!")

        # =========================
        # Step 3: Upload new sample to predict
        # =========================
        st.header("Predict New Sample")
        uploaded_sample = st.file_uploader(
            "Upload CSV for new sample(s) to predict", type="csv", key="new_sample"
        )

        if uploaded_sample:
            new_df = pd.read_csv(uploaded_sample)
            new_df = new_df[feature_cols]

            prediction = clf.predict(new_df)
            probability = clf.predict_proba(new_df)

            class_map = {0: "Healthy", 1: "Parkinson's"}

            for i, pred in enumerate(prediction):
                st.write(f"Sample {i+1}: **Predicted class:** {class_map[pred]}")
                st.write(f"Probability: Healthy={probability[i][0]:.2f}, Parkinson's={probability[i][1]:.2f}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")
