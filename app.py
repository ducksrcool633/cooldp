import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("Parkinson's Spiral Hand Prediction")

# -------------------------------
# Step 1: Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload Spiral Hand CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"CSV loaded with {len(df.columns)} columns.")

        # -------------------------------
        # Step 2: Fix Columns
        # -------------------------------
        expected_cols = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_ET', 'MIN_ET', 'STD_ET',
            'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_ET'
        ]

        # If extra column exists, drop first column
        if len(df.columns) == len(expected_cols) + 1:
            df = df.iloc[:, 1:]

        if len(df.columns) != len(expected_cols):
            st.error(f"CSV has {len(df.columns)} columns but {len(expected_cols)} expected.")
            st.stop()

        # Rename columns to expected
        df.columns = expected_cols

        # -------------------------------
        # Step 3: Check CLASS_TYPE
        # -------------------------------
        if df['CLASS_TYPE'].nunique() < 2:
            st.error("Each class must have at least 2 samples to train the model.")
            st.stop()

        # -------------------------------
        # Step 4: Train Model
        # -------------------------------
        X = df.drop(columns=['ID_PATIENT', 'CLASS_TYPE'])
        y = df['CLASS_TYPE']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)

        # -------------------------------
        # Step 5: Evaluate
        # -------------------------------
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        st.subheader("Model Performance")
        st.write("F1 Score:", round(f1_score(y_test, preds), 3))
        st.write("ROC AUC:", round(roc_auc_score(y_test, probs), 3))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, preds))

        # -------------------------------
        # Step 6: Predict New Sample
        # -------------------------------
        st.subheader("Predict New Sample")
        st.write("Enter values for the following features:")

        new_sample_data = {
            'RMS': st.number_input("RMS", value=0.0),
            'MAX_ET': st.number_input("Max ET", value=0.0),
            'MIN_ET': st.number_input("Min ET", value=0.0),
            'STD_ET': st.number_input("STD ET", value=0.0),
            'MRT': st.number_input("MRT", value=0.0),
            'MAX_HT': st.number_input("Max HT", value=0.0),
            'MIN_HT': st.number_input("Min HT", value=0.0),
            'STD_HT': st.number_input("STD HT", value=0.0),
            'CHANGES_ET': st.number_input("Changes ET", value=0.0)
        }

        if st.button("Predict"):
            sample_df = pd.DataFrame([new_sample_data])
            prediction = clf.predict(sample_df)[0]
            probability = clf.predict_proba(sample_df)[0]

            class_map = {0: "Healthy", 1: "Parkinson's"}
            st.write(f"**Predicted class:** {class_map[prediction]}")
            st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")
else:
    st.info("Please upload a CSV file to start.")
