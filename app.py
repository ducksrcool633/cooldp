# ==============================
# Streamlit App: Parkinson's HandPD Model
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

st.set_page_config(page_title="HandPD Parkinson's Predictor", layout="wide")

st.title("🖊 Parkinson's Hand Tremor Predictor")
st.write("Upload your Spiral HandPD CSV dataset and predict Parkinson's disease from hand tremor features.")

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
uploaded_file = st.file_uploader("Upload Spiral HandPD CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Expected columns
    expected_cols = [
        'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]

    # Check columns
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    # Map CLASS_TYPE: 0 = Healthy, 1 = Parkinson's
    df['CLASS_TYPE'] = df['CLASS_TYPE'].astype(int)

    st.write("### First 5 rows of your dataset")
    st.dataframe(df.head())

    # ------------------------------
    # Step 2: Feature Selection
    # ------------------------------
    feature_cols = expected_cols[2:]  # all numeric features
    X = df[feature_cols]
    y = df['CLASS_TYPE']

    # ------------------------------
    # Step 3: Train Model
    # ------------------------------
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    st.success("Random Forest model trained on uploaded dataset!")

    # ------------------------------
    # Step 4: Show Feature Importance
    # ------------------------------
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    st.write("### Feature Importance")
    st.bar_chart(feat_imp_df.set_index('Feature'))

    # ------------------------------
    # Step 5: Predict New Sample
    # ------------------------------
    st.write("### Predict a New Sample")
    st.write("Enter new feature values to predict Parkinson's:")

    new_data = {}
    for col in feature_cols:
        val = st.number_input(col, value=float(df[col].mean()))
        new_data[col] = [val]

    if st.button("Predict"):
        new_sample = pd.DataFrame(new_data)
        prediction = int(clf.predict(new_sample)[0])
        probability = clf.predict_proba(new_sample)[0]
        class_map = {0: "Healthy", 1: "Parkinson's"}

        st.write(f"**Predicted class:** {class_map[prediction]}")
        st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

    # ------------------------------
    # Step 6: Boxplot Dashboard
    # ------------------------------
    st.write("### Boxplot Dashboard")
    for feature in feature_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x='CLASS_TYPE', y=feature, data=df, showfliers=True, ax=ax)
        means = df.groupby('CLASS_TYPE')[feature].mean().values
        for cls, mean_val in enumerate(means):
            ax.scatter(cls, mean_val, color='red', marker='D', s=50, label='Mean' if cls==0 else "")
        ax.set_xticklabels(['Healthy', "Parkinson's"])
        ax.set_title(f"{feature} by Class")
        ax.set_ylabel(feature)
        ax.set_xlabel("Class")
        st.pyplot(fig)

    # ------------------------------
    # Step 7: RMS vs MRT Scatterplot
    # ------------------------------
    st.write("### RMS vs MRT Scatterplot")
    fig, ax = plt.subplots()
    sns.scatterplot(x='RMS', y='MRT', hue='CLASS_TYPE', data=df, palette=['green', 'orange'], s=70, ax=ax)
    ax.set_xlabel("RMS (magnitude of pen movement)")
    ax.set_ylabel("MRT (mean relative tremor)")
    ax.set_title("RMS vs MRT by Class")
    ax.legend(labels=['Healthy', "Parkinson's"])
    st.pyplot(fig)

st.info("✅ App ready! Upload a CSV, explore features, train model, and predict new samples.")
