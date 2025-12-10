# ==============================
# HandPD Streamlit App - Cleaned for Training
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

st.title("🖐 HandPD Parkinson's Detection (Cleaned)")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload Spiral_HandPD CSV", type="csv")

if uploaded_file is not None:
    # Try loading CSV
    try:
        spiral_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
    
    # --- Step 2: Fix column names ---
    expected_cols = [
        'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 
        'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT', 
        'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    if not all(col in spiral_df.columns for col in expected_cols):
        st.warning("CSV headers missing or incorrect. Fixing automatically...")
        spiral_df = pd.read_csv(uploaded_file, names=expected_cols, header=0)
    
    st.write("Columns detected:", list(spiral_df.columns))
    
    # --- Step 3: Convert features to numeric and handle NaN ---
    feature_cols = expected_cols[2:]  # all numeric features
    spiral_df[feature_cols] = spiral_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    spiral_df[feature_cols] = spiral_df[feature_cols].fillna(0)
    
    # Map CLASS_TYPE
    spiral_df['CLASS_TYPE'] = pd.to_numeric(spiral_df['CLASS_TYPE'], errors='coerce').fillna(0).astype(int)
    spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1  # 0=Healthy, 1=Parkinson's
    
    # --- Step 4: Train Random Forest ---
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    st.success("Model trained successfully!")
    
    # --- Step 5: Predict new sample ---
    st.subheader("Predict New Sample")
    new_sample_data = {}
    for feature in feature_cols:
        new_sample_data[feature] = [st.number_input(f"{feature}:", value=0.0)]
    
    if st.button("Predict"):
        new_sample_df = pd.DataFrame(new_sample_data)
        prediction = clf.predict(new_sample_df)[0]
        prob = clf.predict_proba(new_sample_df)[0]
        class_map = {0: "Healthy", 1: "Parkinson's"}
        
        st.write(f"**Predicted Class:** {class_map[prediction]}")
        st.write(f"**Probability:** Healthy={prob[0]:.2f}, Parkinson's={prob[1]:.2f}")
    
    # --- Step 6: Feature Importance ---
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    st.subheader("Feature Importance")
    st.bar_chart(feat_df.set_index('Feature'))
    
    st.success("✅ App ready! Upload CSV, explore data, train model, and predict new samples.")
