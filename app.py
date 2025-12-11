import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Parkinson's Hand Tremor Predictor")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Base expected column names
        expected_cols = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT',
            'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT', 'MRT',
            'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_ET_HT'
        ]

        # Adjust length: truncate or pad with generic names
        if len(df.columns) > len(expected_cols):
            st.warning(f"CSV has {len(df.columns)} columns, more than expected. Extra columns will be ignored.")
            df.columns = expected_cols + [f"EXTRA_{i}" for i in range(len(df.columns)-len(expected_cols))]
        elif len(df.columns) < len(expected_cols):
            st.warning(f"CSV has {len(df.columns)} columns, fewer than expected. Padding missing column names.")
            df.columns = list(df.columns) + expected_cols[len(df.columns):]
        else:
            df.columns = expected_cols

        # Ensure numeric columns are numeric
        numeric_cols = expected_cols[2:]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        # Check class counts
        class_counts = df['CLASS_TYPE'].value_counts()
        if (class_counts < 2).any():
            st.error(f"Each class must have at least 2 samples. Counts: {class_counts.to_dict()}")
        else:
            st.success("CSV loaded successfully!")

            # Train model
            X = df[numeric_cols]
            y = df['CLASS_TYPE']
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X, y)
            st.success("Model trained!")

            # Predict new sample
            st.subheader("Predict New Sample")
            input_data = {}
            for col in numeric_cols:
                input_data[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                sample_df = pd.DataFrame([input_data])
                pred_class = clf.predict(sample_df)[0]
                pred_prob = clf.predict_proba(sample_df)[0]
                class_map = {0: "Healthy", 1: "Parkinson's"}
                st.write(f"**Predicted class:** {class_map.get(pred_class, pred_class)}")
                st.write(f"**Probability:** Healthy={pred_prob[0]:.2f}, Parkinson's={pred_prob[1]:.2f}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")
else:
    st.info("Please upload a CSV file to get started.")
