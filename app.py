# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("Parkinson's Hand Tremor Prediction")

# --- Step 1: Upload Dataset ---
st.info("Upload the dataset CSV (Spiral_HandPD.csv) to train the model.")
data_file = st.file_uploader("Upload training CSV", type=["csv"])

if data_file is not None:
    try:
        df = pd.read_csv(data_file)
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    # Check expected columns
    expected_cols = [
        'ID_PATIENT','CLASS_TYPE','RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT','MRT','MAX_HT','MIN_HT','STD_HT',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    
    if not all(col in df.columns for col in expected_cols):
        st.error(f"CSV missing required columns! Found: {list(df.columns)}")
        st.stop()
    
    # Map CLASS_TYPE: 0 = Healthy, 1 = Parkinson's
    df['CLASS_TYPE'] = df['CLASS_TYPE'].astype(int) - 1
    
    # Features & labels
    feature_cols = expected_cols[2:]  # All columns except ID and CLASS_TYPE
    X = df[feature_cols]
    y = df['CLASS_TYPE']
    
    # Train model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    st.success("Model trained successfully!")
    
    # --- Step 2: Predict new sample ---
    st.header("Predict a new sample")
    st.info("Upload a CSV with a single row containing the same features as training data (without ID_PATIENT or CLASS_TYPE).")
    sample_file = st.file_uploader("Upload new sample CSV", type=["csv"], key="sample")
    
    if sample_file is not None:
        try:
            new_sample = pd.read_csv(sample_file)
            
            # Ensure correct columns
            if not all(col in feature_cols for col in new_sample.columns):
                st.error(f"New sample CSV must have columns: {feature_cols}")
                st.stop()
            
            prediction = clf.predict(new_sample)[0]
            probability = clf.predict_proba(new_sample)[0]
            class_map = {0: "Healthy", 1: "Parkinson's"}
            
            st.write(f"**Predicted class:** {class_map[prediction]}")
            st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")
        except Exception as e:
            st.error(f"Error processing new sample: {e}")
