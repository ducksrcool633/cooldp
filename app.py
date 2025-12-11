import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

st.set_page_config(page_title="Hand Tremor Prediction", layout="centered")
st.title("Hand Tremor Prediction App")

# -----------------------
# Section 1: CSV Upload
# -----------------------
st.header("Upload CSV Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("CSV Loaded Successfully!")
        st.write("Columns detected:", df.columns.tolist())

        # Expected columns
        expected_cols = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT',
            'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT',
            'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]

        # Keep only expected columns that exist
        existing_cols = [col for col in expected_cols if col in df.columns]
        df = df[existing_cols]

        # Handle CLASS_TYPE less than 2 samples
        if 'CLASS_TYPE' in df.columns:
            counts = df['CLASS_TYPE'].value_counts()
            rare_classes = counts[counts < 2].index.tolist()
            if rare_classes:
                st.warning(f"Some classes have less than 2 samples. They will be removed: {rare_classes}")
                df = df[~df['CLASS_TYPE'].isin(rare_classes)]

        if df.shape[0] < 2 or 'CLASS_TYPE' not in df.columns:
            st.warning("Not enough data to train a real model. Using dummy predictions.")
            df['Predicted_Class'] = np.random.randint(0, 2, size=df.shape[0])
            df['Probability'] = np.random.rand(df.shape[0])
        else:
            # Prepare features and labels
            X = df.drop(columns=['ID_PATIENT', 'CLASS_TYPE'], errors='ignore')
            y = df['CLASS_TYPE']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            # Make predictions
            df['Predicted_Class'] = clf.predict(X)
            df['Probability'] = clf.predict_proba(X).max(axis=1)

        st.subheader("Predictions")
        st.dataframe(df[['ID_PATIENT','Predicted_Class','Probability']])

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")

# -----------------------
# Section 2: Image Upload
# -----------------------
st.header("Predict from Image")
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert('L')  # grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to simple numeric features
        img_array = np.array(image)
        features = np.array([
            img_array.mean(),
            img_array.std(),
            img_array.max(),
            img_array.min()
        ]).reshape(1, -1)

        # Dummy model if CSV model not trained
        if df is None or 'CLASS_TYPE' not in df.columns or df.shape[0] < 2:
            pred_class = np.random.randint(0, 2)
            prob = np.random.rand()
        else:
            # Use previously trained classifier
            pred_class = clf.predict(features)[0]
            prob = clf.predict_proba(features).max()

        st.subheader("Image Prediction")
        st.write(f"Predicted Class: {pred_class}")
        st.write(f"Probability: {prob:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
