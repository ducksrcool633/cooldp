import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Parkinson's HandPD Predictor")
st.write("Upload a CSV file with spiral handwriting features to predict Parkinson's.")

# ------------------------------
# Step 1: Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
if uploaded_file is not None:
    try:
        # Read CSV without headers
        spiral_df = pd.read_csv(uploaded_file, header=None)
        st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# ------------------------------
# Step 2: Assign expected columns
# ------------------------------
expected_cols = [
    "ID_PATIENT","CLASS_TYPE","RMS","MAX_BETWEEN_ET_HT","MIN_BETWEEN_ET_HT",
    "STD_DEVIATION_ET_HT","MRT","MAX_HT","MIN_HT","STD_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT"
]

if spiral_df.shape[1] != len(expected_cols):
    st.error(f"CSV has {spiral_df.shape[1]} columns but {len(expected_cols)} were expected.")
    st.stop()

spiral_df.columns = expected_cols

# ------------------------------
# Step 3: Preprocess data
# ------------------------------
spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1

feature_cols = [
    'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT',
    'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
]

X = spiral_df[feature_cols]
y = spiral_df['CLASS_TYPE']

# ------------------------------
# Step 4: Train model
# ------------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)
st.success("Model trained successfully!")

# ------------------------------
# Step 5: Predict new sample
# ------------------------------
st.subheader("Predict a new sample")
st.write("Enter feature values below:")

new_sample_data = {}
for col in feature_cols:
    new_sample_data[col] = [st.number_input(col, value=float(spiral_df[col].mean()))]

new_sample = pd.DataFrame(new_sample_data)

if st.button("Predict"):
    prediction = clf.predict(new_sample)[0]
    probability = clf.predict_proba(new_sample)[0]
    class_map = {0: "Healthy", 1: "Parkinson's"}
    st.write(f"**Predicted class:** {class_map[prediction]}")
    st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")
