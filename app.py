import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

st.title("HandPD Parkinson's Prediction App 🖐️")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload Spiral_HandPD CSV file", type="csv")

if uploaded_file is not None:
    # --- Step 2: Force column names ---
    expected_cols = [
        'ID_PATIENT',
        'CLASS_TYPE',
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
    
    spiral_df = pd.read_csv(uploaded_file, header=None)
    
    if spiral_df.shape[1] != len(expected_cols):
        st.error(f"CSV has {spiral_df.shape[1]} columns but {len(expected_cols)} expected!")
        st.stop()
    
    spiral_df.columns = expected_cols

    # --- Step 3: Prepare data ---
    spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1  # 0 = Healthy, 1 = Parkinson's
    feature_cols = expected_cols[2:]  # all numeric features
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']

    # --- Step 4: Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Step 5: Train model ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # --- Step 6: Evaluate model ---
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    st.subheader("Model Performance on Test Set")
    st.write("F1 Score:", round(f1_score(y_test, preds), 3))
    st.write("ROC AUC:", round(roc_auc_score(y_test, probs), 3))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, preds))

    # --- Step 7: Predict new sample ---
    st.subheader("Predict a New Sample")
    st.write("Enter values for each feature:")

    new_sample = pd.DataFrame({
        'RMS': [st.number_input("RMS", value=1200)],
        'MAX_BETWEEN_ET_HT': [st.number_input("MAX_BETWEEN_ET_HT", value=30)],
        'MIN_BETWEEN_ET_HT': [st.number_input("MIN_BETWEEN_ET_HT", value=5)],
        'STD_DEVIATION_ET_HT': [st.number_input("STD_DEVIATION_ET_HT", value=12)],
        'MRT': [st.number_input("MRT", value=0.8)],
        'MAX_HT': [st.number_input("MAX_HT", value=150)],
        'MIN_HT': [st.number_input("MIN_HT", value=20)],
        'STD_HT': [st.number_input("STD_HT", value=25)],
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT': [
            st.number_input("CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT", value=10)
        ]
    })

    if st.button("Predict"):
        prediction = clf.predict(new_sample)
        probability = clf.predict_proba(new_sample)[0]

        class_map = {0: "Healthy", 1: "Parkinson's"}
        st.write(f"**Predicted class:** {class_map[int(prediction[0])]}")
        st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

    st.success("✅ App is ready! Upload CSV and try predictions.")

else:
    st.info("Please upload the Spiral_HandPD CSV file to start.")
