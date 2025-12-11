# cooldp_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("Parkinson's Hand Tremor Predictor")
st.write("Upload a CSV file of hand tremor measurements to predict Parkinson's probabilities.")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Simplify column names for ease
        expected_cols = [
            'ID', 'CLASS', 'RMS', 'MAX_ET', 'MIN_ET', 'STD_ET',
            'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_ET'
        ]
        if len(df.columns) != len(expected_cols):
            st.error(f"CSV has {len(df.columns)} columns but {len(expected_cols)} expected.")
        else:
            df.columns = expected_cols
            
            st.write("CSV loaded successfully! First 5 rows:")
            st.dataframe(df.head())
            
            # --- Step 2: Ensure each class has at least 2 samples ---
            for cls in df['CLASS'].unique():
                count = len(df[df['CLASS'] == cls])
                if count < 2:
                    df = pd.concat([df, df[df['CLASS'] == cls].sample(2-count, replace=True)])
            
            # --- Step 3: Train/Test split ---
            X = df[['RMS', 'MAX_ET', 'MIN_ET', 'STD_ET', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_ET']]
            y = df['CLASS']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # --- Step 4: Train RandomForest ---
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_train, y_train)
            
            # --- Step 5: Evaluate Model ---
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)
            st.write("Model trained successfully!")
            st.write(f"F1 Score: {f1_score(y_test, preds, average='macro'):.3f}")
            
            # --- Step 6: Predict new data ---
            st.subheader("Predict New Samples")
            new_file = st.file_uploader("Upload new CSV to predict", type="csv", key="new")
            
            if new_file:
                new_df = pd.read_csv(new_file)
                if new_df.shape[1] != 9:
                    st.error("New CSV must have exactly 9 numeric columns: RMS, MAX_ET, MIN_ET, STD_ET, MRT, MAX_HT, MIN_HT, STD_HT, CHANGES_ET")
                else:
                    new_df.columns = ['RMS', 'MAX_ET', 'MIN_ET', 'STD_ET', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_ET']
                    predictions = clf.predict(new_df)
                    probabilities = clf.predict_proba(new_df)
                    
                    st.write("Predictions and probabilities:")
                    for i, pred in enumerate(predictions):
                        prob_str = ", ".join([f"Class {cls}={prob:.2f}" for cls, prob in zip(clf.classes_, probabilities[i])])
                        st.write(f"Sample {i+1}: Predicted Class={pred}, Probabilities=({prob_str})")
                    
    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")
