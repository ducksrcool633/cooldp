import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("Parkinson's Spiral Hand Prediction")

# ===========================
# Upload CSV
# ===========================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    # ===========================
    # Check columns
    # ===========================
    expected_cols = [
        'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT',
        'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT',
        'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    
    if not all(col in df.columns for col in expected_cols):
        st.error(f"CSV missing required columns! Found: {list(df.columns)}")
        st.stop()
    
    # ===========================
    # Convert to numeric
    # ===========================
    df_numeric = df.copy()
    for col in expected_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Drop rows with NaN
    df_numeric.dropna(inplace=True)
    
    # ===========================
    # Check sample size
    # ===========================
    counts = df_numeric['CLASS_TYPE'].value_counts()
    if (counts < 2).any():
        st.error(f"Each class must have at least 2 samples. Counts: {counts.to_dict()}")
        st.stop()
    
    # ===========================
    # Split features and labels
    # ===========================
    feature_cols = [
        'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT',
        'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    
    X = df_numeric[feature_cols]
    y = df_numeric['CLASS_TYPE']
    
    # ===========================
    # Train model
    # ===========================
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.stop()
    
    st.success("Model trained successfully!")
    
    # ===========================
    # Prediction feature
    # ===========================
    st.header("Predict New Sample")
    
    new_sample_dict = {}
    for feature in feature_cols:
        new_sample_dict[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("Predict"):
        new_sample = pd.DataFrame([new_sample_dict])
        prediction = clf.predict(new_sample)[0]
        probability = clf.predict_proba(new_sample)[0]
        class_map = {0: "Healthy", 1: "Parkinson's"}
        st.write(f"**Predicted class:** {class_map.get(prediction, prediction)}")
        st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")
