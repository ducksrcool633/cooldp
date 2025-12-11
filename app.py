# ==============================================
# Streamlit App: Parkinson's Spiral HandPD Predictor
# Fully self-contained and safe for CSV uploads
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Parkinson's HandPD Predictor", layout="wide")

st.title("Parkinson's Spiral HandPD Predictor")
st.write("Upload your CSV dataset to train the model and predict new samples.")

# -------------------------------
# Expected columns
# -------------------------------
expected_cols = [
    'ID_PATIENT','CLASS_TYPE','RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT',
    'STD_DEVIATION_ET_HT','MRT','MAX_HT','MIN_HT','STD_HT',
    'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
]

# -------------------------------
# Upload CSV
# -------------------------------
data_file = st.file_uploader("Upload your dataset CSV", type=["csv"])

if data_file is not None:
    try:
        df = pd.read_csv(data_file, header=None)  # read without assuming header
        if df.shape[1] < len(expected_cols):
            st.error(f"CSV has too few columns. Expected at least {len(expected_cols)} columns.")
        else:
            # Only keep the first expected_cols length columns
            df = df.iloc[:, :len(expected_cols)]
            df.columns = expected_cols

            st.success("CSV loaded successfully!")

            # Show basic dataset info
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            st.subheader("Class Distribution")
            st.bar_chart(df['CLASS_TYPE'].value_counts())

            # -------------------------------
            # Train/Test Split
            # -------------------------------
            numeric_features = expected_cols[2:]  # All numeric columns
            X = df[numeric_features]
            y = df['CLASS_TYPE']

            # Simple check: need at least 2 samples per class
            if min(y.value_counts()) < 2:
                st.error("Each class must have at least 2 samples to train the model.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # -------------------------------
                # Train Random Forest Model
                # -------------------------------
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
                clf.fit(X_train, y_train)

                # -------------------------------
                # Evaluate Model
                # -------------------------------
                preds = clf.predict(X_test)
                probs = clf.predict_proba(X_test)[:,1]

                st.subheader("Model Evaluation")
                st.write(f"F1 Score: {f1_score(y_test, preds):.3f}")
                st.write(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, preds))

                # -------------------------------
                # Predict New Sample
                # -------------------------------
                st.subheader("Predict New Sample")
                st.write("Upload a new CSV with the same numeric columns (no CLASS_TYPE needed).")

                sample_file = st.file_uploader("Upload new sample CSV", type=["csv"], key="sample")

                if sample_file is not None:
                    try:
                        sample_df = pd.read_csv(sample_file, header=None)
                        # Keep only the numeric feature columns
                        sample_df = sample_df.iloc[:, :len(numeric_features)]
                        sample_df.columns = numeric_features

                        pred = clf.predict(sample_df)
                        prob = clf.predict_proba(sample_df)

                        class_map = {0: "Healthy", 1: "Parkinson's"}

                        st.write("### Predictions")
                        for i in range(len(sample_df)):
                            st.write(f"Sample {i+1}:")
                            st.write(f"Predicted class: {class_map[pred[i]]}")
                            st.write(f"Probability: Healthy={prob[i][0]:.2f}, Parkinson's={prob[i][1]:.2f}")

                    except Exception as e:
                        st.error(f"Error reading new sample CSV: {e}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")

else:
    st.info("Please upload a CSV to continue.")
