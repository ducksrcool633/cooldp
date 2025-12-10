# ==============================
# Streamlit HandPD App (Fixed & Plug-and-Play)
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("Parkinson's Spiral HandPD Predictor ✅")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload your Spiral_HandPD CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    spiral_df = pd.read_csv(uploaded_file)
    
    # --- Step 2: Fix columns ---
    expected_cols = [
        "ID_PATIENT",
        "CLASS_TYPE",
        "RMS",
        "MAX_BETWEEN_ET_HT",
        "MIN_BETWEEN_ET_HT",
        "STD_DEVIATION_ET_HT",
        "MRT",
        "MAX_HT",
        "MIN_HT",
        "STD_HT",
        "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT"
    ]
    
    if len(spiral_df.columns) != len(expected_cols):
        st.warning(f"CSV has {len(spiral_df.columns)} columns; trimming to first {len(expected_cols)} columns.")
        spiral_df = spiral_df.iloc[:, :len(expected_cols)]
    
    spiral_df.columns = expected_cols
    
    # --- Step 3: Map CLASS_TYPE ---
    if "CLASS_TYPE" in spiral_df.columns:
        spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1
    else:
        st.error("CSV does not contain 'CLASS_TYPE' column!")
    
    st.success("CSV loaded and cleaned successfully!")
    
    # --- Step 4: Show basic info ---
    st.subheader("Dataset Preview")
    st.dataframe(spiral_df.head())
    
    st.subheader("Class Distribution")
    st.bar_chart(spiral_df['CLASS_TYPE'].value_counts())
    
    # --- Step 5: Feature Selection ---
    feature_cols = [
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
    
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']
    
    # --- Step 6: Train/Test Split & Model ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # --- Step 7: Model Performance ---
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    
    st.subheader("Model Performance on Test Set")
    st.write("F1 Score:", round(f1_score(y_test, preds), 3))
    st.write("ROC AUC:", round(roc_auc_score(y_test, probs), 3))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, preds))
    
    # --- Step 8: Feature Importance ---
    importances = clf.feature_importances_
    feat_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)
    
    st.subheader("Feature Importance")
    st.bar_chart(feat_importance_df.set_index('feature'))
    
    # --- Step 9: Predict New Sample ---
    st.subheader("Predict New Sample")
    st.write("Enter feature values for a new drawing:")
    
    new_sample = {}
    for feature in feature_cols:
        new_sample[feature] = st.number_input(feature, value=float(0))
    
    if st.button("Predict"):
        new_df = pd.DataFrame([new_sample])
        prediction = clf.predict(new_df)
        probability = clf.predict_proba(new_df)
        class_map = {0: "Healthy", 1: "Parkinson's"}
        
        st.write("Predicted class:", class_map[prediction[0]])
        st.write(f"Probability: Healthy={probability[0][0]:.2f}, Parkinson's={probability[0][1]:.2f}")
