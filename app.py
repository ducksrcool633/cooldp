# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from io import BytesIO

st.title("Parkinson's Hand Tremor Predictor")

# -------------------------------
# Step 1: Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Expected columns
        expected_cols = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT',
            'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT',
            'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]

        if df.shape[1] != len(expected_cols):
            st.warning(f"CSV has {df.shape[1]} columns but {len(expected_cols)} expected. Renaming automatically.")
            df.columns = expected_cols[:df.shape[1]]

        # Check numeric column
        numeric_cols = expected_cols[2:]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with missing values
        df = df.dropna()

        # Check class distribution
        class_counts = df['CLASS_TYPE'].value_counts()
        if (class_counts < 2).any():
            st.error(f"Each class must have at least 2 samples. Current counts:\n{class_counts.to_dict()}")
        else:
            st.success("CSV loaded successfully! Ready to train model.")

            # -------------------------------
            # Step 2: Train model
            # -------------------------------
            feature_cols = numeric_cols
            X = df[feature_cols]
            y = df['CLASS_TYPE']

            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X, y)

            st.success("Model trained successfully!")

            # -------------------------------
            # Step 3: Predict new sample
            # -------------------------------
            st.subheader("Predict New Sample")
            st.write("Enter values for a new drawing:")

            input_data = {}
            for col in feature_cols:
                input_data[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                sample_df = pd.DataFrame([input_data])
                pred_class = clf.predict(sample_df)[0]
                pred_prob = clf.predict_proba(sample_df)[0]

                class_map = {0: "Healthy", 1: "Parkinson's"}
                st.write(f"**Predicted class:** {class_map.get(pred_class, pred_class)}")
                st.write(f"**Probability:** Healthy={pred_prob[0]:.2f}, Parkinson's={pred_prob[1]:.2f}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")

else:
    st.info("Please upload a CSV file to get started.")
