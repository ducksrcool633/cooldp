import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Hand Tremor Prediction App")

# Expected columns
expected_cols = [
    'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT',
    'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT',
    'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
]

st.header("Upload CSV or Image")

uploaded_file = st.file_uploader("Upload CSV or image", type=["csv","png","jpg","jpeg"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

            # Automatically rename columns if count matches expected
            if df.shape[1] == len(expected_cols):
                df.columns = expected_cols
            else:
                st.error(f"CSV has {df.shape[1]} columns but {len(expected_cols)} expected. Please check your file.")
                st.stop()

            # Check required columns
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                st.error(f"CSV missing required columns: {missing_cols}")
                st.stop()

            # Check class counts
            class_counts = df['CLASS_TYPE'].value_counts()
            low_classes = class_counts[class_counts < 2]
            if not low_classes.empty:
                st.warning(f"Some classes have less than 2 samples. They will be removed: {list(low_classes.index)}")
                df = df[~df['CLASS_TYPE'].isin(low_classes.index)]

            X = df.drop(['ID_PATIENT', 'CLASS_TYPE'], axis=1)
            y = df['CLASS_TYPE']

            # Encode if needed
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Train simple model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)

            st.success("Model trained on your CSV data!")

            st.subheader("Predictions on uploaded CSV")
            y_pred = clf.predict(X)
            y_prob = clf.predict_proba(X)

            pred_df = pd.DataFrame({
                "Patient_ID": df['ID_PATIENT'],
                "Predicted_Class": y_pred,
                "Probability": [np.max(p) for p in y_prob]
            })
            st.dataframe(pred_df)

        else:
            # Placeholder for image conversion
            st.info("Image uploaded. Currently, image conversion to features is simulated.")

            # Simulate feature extraction
            dummy_features = np.random.rand(1, 9)  # 9 features matching numeric columns
            dummy_X = pd.DataFrame(dummy_features, columns=[
                'RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT',
                'STD_DEVIATION_ET_HT','MRT','MAX_HT','MIN_HT','STD_HT',
                'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
            ])

            # Train dummy model on dummy data
            dummy_y = np.array([0,1,0,1,0,1,0,1,0,1])
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(dummy_X, dummy_y[:dummy_X.shape[0]])  # train on subset

            pred_class = clf.predict(dummy_X)[0]
            pred_prob = clf.predict_proba(dummy_X).max()

            st.write(f"**Predicted class:** {pred_class}")
            st.write(f"**Probability:** {pred_prob:.2f}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")
