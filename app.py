# ==============================
# Streamlit App for HandPD Parkinson's Prediction
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import joblib

# --- Step 1: Upload CSV ---
st.title("HandPD Parkinson's Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Spiral_HandPD.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Clean column names
    df.columns = [c.strip().upper() for c in df.columns]  # remove spaces & uppercase
    st.write("Columns detected in CSV:", df.columns.tolist())
    
    # Find the CLASS_TYPE column (case insensitive)
    class_col_candidates = [c for c in df.columns if "CLASS_TYPE" in c]
    if not class_col_candidates:
        st.error("No column containing 'CLASS_TYPE' found in CSV!")
        st.stop()
    class_col = class_col_candidates[0]
    
    # Map CLASS_TYPE: 0=Healthy, 1=Parkinson's
    df[class_col] = df[class_col] - 1
    
    # Define numeric features
    numeric_features = [
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
    
    # Automatically find numeric features in CSV
    numeric_features_in_df = [f for f in numeric_features if f in df.columns]
    
    if not numeric_features_in_df:
        st.error("No numeric feature columns found in CSV!")
        st.stop()
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # --- Step 2: Train/Test Split & Random Forest ---
    X = df[numeric_features_in_df]
    y = df[class_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    
    st.subheader("Model Performance on Test Set")
    st.write(f"F1 Score: {f1_score(y_test, preds):.3f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, preds))
    
    # --- Step 3: Predict New Sample ---
    st.subheader("Predict New Sample")
    st.write("Enter new feature values below:")
    
    new_sample = {}
    for feature in numeric_features_in_df:
        val = st.number_input(f"{feature}", value=float(df[feature].mean()))
        new_sample[feature] = [val]
    
    if st.button("Predict"):
        sample_df = pd.DataFrame(new_sample)
        prediction = clf.predict(sample_df)
        probability = clf.predict_proba(sample_df)
        class_map = {0: "Healthy", 1: "Parkinson's"}
        st.success(f"Predicted class: {class_map[prediction[0]]}")
        st.info(f"Probability: Healthy={probability[0][0]:.2f}, Parkinson's={probability[0][1]:.2f}")
    
    # Save model for later use
    joblib.dump(clf, "handpd_model.pkl")
else:
    st.info("Please upload the Spiral_HandPD.csv file to proceed.")


