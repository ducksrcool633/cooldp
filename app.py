# ==============================
# HandPD Streamlit App - Auto-Detect CSV Headers
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("🖐 HandPD Parkinson's Detection")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload Spiral_HandPD CSV", type="csv")

if uploaded_file is not None:
    # Try loading CSV with headers
    try:
        spiral_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
    
    # --- Step 2: Check columns ---
    required_columns = [
        'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 
        'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT', 
        'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    
    # If columns not found, assign headers
    if not all(col in spiral_df.columns for col in required_columns):
        st.warning("CSV headers missing or incorrect. Automatically fixing...")
        spiral_df = pd.read_csv(uploaded_file, names=required_columns, header=0)
    
    st.write("Columns in dataset:", list(spiral_df.columns))
    
    # --- Step 3: Map CLASS_TYPE ---
    spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1  # 0=Healthy, 1=Parkinson's
    
    # --- Step 4: Show basic info ---
    st.subheader("Dataset Overview")
    st.write(spiral_df.head())
    st.write("Class distribution:")
    st.write(spiral_df['CLASS_TYPE'].value_counts().rename({0:"Healthy", 1:"Parkinson's"}))
    
    # --- Step 5: Feature Selection ---
    feature_cols = [
        'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 
        'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']
    
    # --- Step 6: Train Model ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    
    st.success("Random Forest model trained successfully!")
    
    # --- Step 7: Predict New Sample ---
    st.subheader("Predict New Sample")
    new_sample_data = {}
    for feature in feature_cols:
        val = st.number_input(f"{feature}:", value=float(0))
        new_sample_data[feature] = [val]
    
    if st.button("Predict"):
        new_sample_df = pd.DataFrame(new_sample_data)
        pred = clf.predict(new_sample_df)[0]
        prob = clf.predict_proba(new_sample_df)[0]
        class_map = {0: "Healthy", 1: "Parkinson's"}
        
        st.write(f"**Predicted Class:** {class_map[pred]}")
        st.write(f"**Probability:** Healthy={prob[0]:.2f}, Parkinson's={prob[1]:.2f}")
    
    # --- Step 8: Feature Importance ---
    st.subheader("Feature Importance")
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(feat_df.set_index('Feature'))

    st.success("✅ All done! Upload CSV, explore data, train model, and predict new samples.")
