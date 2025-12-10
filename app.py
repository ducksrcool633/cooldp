# ==============================
# Streamlit HandPD Predictor
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import joblib

st.title("🖊 Parkinson's Spiral HandPD Predictor")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload Spiral_HandPD CSV", type=["csv"])
if uploaded_file is not None:
    
    # --- Step 2: Fix headers if necessary ---
    expected_cols = [
        'ID_PATIENT','CLASS_TYPE','RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT','MRT','MAX_HT','MIN_HT','STD_HT',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    try:
        spiral_df = pd.read_csv(uploaded_file, header=0)
        if list(spiral_df.columns) != expected_cols:
            spiral_df = pd.read_csv(uploaded_file, header=None)
            spiral_df.columns = expected_cols
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    st.success("CSV loaded successfully!")

    # --- Step 3: Show first few rows ---
    st.subheader("Dataset Preview")
    st.dataframe(spiral_df.head())

    # --- Step 4: Define features & labels ---
    feature_cols = [
        'RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT','STD_DEVIATION_ET_HT',
        'MRT','MAX_HT','MIN_HT','STD_HT','CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']

    # --- Step 5: Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Step 6: Train Random Forest ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # --- Step 7: Evaluate model ---
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    st.subheader("Model Performance")
    st.write("F1 Score:", round(f1_score(y_test, preds),3))
    st.write("ROC AUC:", round(roc_auc_score(y_test, probs),3))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, preds))

    # --- Step 8: Feature Importance ---
    st.subheader("Feature Importance")
    importances = clf.feature_importances_
    feat_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)
    st.bar_chart(feat_importance_df.set_index('feature'))

    # --- Step 9: Dashboard Plots ---
    st.subheader("Feature Distributions by Class")
    for feature in feature_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True, ax=ax)
        sns.pointplot(x='CLASS_TYPE', y=feature, data=spiral_df, estimator=np.mean,
                      color='red', markers='D', scale=1.0, errorbar=None, ax=ax)
        ax.set_xticklabels(['Healthy', "Parkinson's"])
        ax.set_title(f"{feature} by Class")
        st.pyplot(fig)

    # RMS vs MRT Scatter
    st.subheader("RMS vs MRT Scatterplot")
    fig, ax = plt.subplots()
    sns.scatterplot(x='RMS', y='MRT', hue='CLASS_TYPE', data=spiral_df, palette=['green','orange'], s=70, ax=ax)
    ax.set_xlabel("RMS (magnitude of pen movement)")
    ax.set_ylabel("MRT (mean relative tremor)")
    ax.set_title("RMS vs MRT by Class")
    ax.legend(labels=['Healthy','Parkinson\'s'])
    st.pyplot(fig)

    # --- Step 10: Predict New Sample ---
    st.subheader("Predict a New Sample")
    new_input = {}
    for feature in feature_cols:
        new_input[feature] = st.number_input(f"{feature}:", value=float(spiral_df[feature].mean()))
    
    if st.button("Predict"):
        new_sample = pd.DataFrame([new_input])
        prediction = clf.predict(new_sample)
        probability = clf.predict_proba(new_sample)[0]
        class_map = {0:"Healthy", 1:"Parkinson's"}
        st.success(f"Predicted class: {class_map[prediction[0]]}")
        st.info(f"Probability → Healthy: {probability[0]:.2f}, Parkinson's: {probability[1]:.2f}")

    # --- Step 11: Save Model ---
    joblib.dump(clf, "handpd_model.pkl")
    st.info("Model saved as handpd_model.pkl")
