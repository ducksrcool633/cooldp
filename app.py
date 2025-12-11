# ==============================
# Streamlit App: Parkinson's Spiral Hand Model
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

st.title("🖊️ Parkinson's Hand Drawing Predictor")
st.write("Upload a CSV of spiral hand drawings to predict if someone has Parkinson's.")

# ------------------------------
# Step 1: Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Please upload a CSV to continue.")
    st.stop()

# ------------------------------
# Step 2: Clean CSV
# ------------------------------
expected_cols = [
    'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
    'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
    'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
]

if len(df.columns) != len(expected_cols):
    st.warning(f"CSV has {len(df.columns)} columns but {len(expected_cols)} expected.")
    st.write("Columns found:", list(df.columns))
    st.stop()

df.columns = expected_cols

# Map CLASS_TYPE to 0=Healthy, 1=Parkinson's if needed
df['CLASS_TYPE'] = df['CLASS_TYPE'].apply(lambda x: 0 if x==0 else 1)

# ------------------------------
# Step 3: Define Features
# ------------------------------
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

# Kid-friendly labels
feature_names = {
    'RMS': 'Hand Shake Size',
    'MAX_BETWEEN_ET_HT': 'Biggest Time Gap',
    'MIN_BETWEEN_ET_HT': 'Smallest Time Gap',
    'STD_DEVIATION_ET_HT': 'Time Gap Variation',
    'MRT': 'Average Tremor Speed',
    'MAX_HT': 'Biggest Hand Lift',
    'MIN_HT': 'Smallest Hand Lift',
    'STD_HT': 'Hand Lift Variation',
    'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT': 'Changes in Hand Movement'
}

X = df[feature_cols]
y = df['CLASS_TYPE']

# ------------------------------
# Step 4: Train/Test Split & Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

st.subheader("📊 Model Performance")
st.write(f"F1 Score: {round(f1_score(y_test, preds), 2)}")
st.write(f"ROC AUC: {round(roc_auc_score(y_test, probs), 2)}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, preds))

# ------------------------------
# Step 5: Feature Importance
# ------------------------------
importances = clf.feature_importances_
feat_df = pd.DataFrame({'Feature': [feature_names[f] for f in feature_cols], 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8,5))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance (Kid-Friendly)")
st.pyplot(plt)

# ------------------------------
# Step 6: Plot simple boxplots
# ------------------------------
st.subheader("📈 Feature Comparison by Class")
for col in feature_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='CLASS_TYPE', y=col, data=df, palette=['green','orange'])
    plt.xticks([0,1], ['Healthy','Parkinson\'s'])
    plt.ylabel(feature_names[col])
    plt.title(f"{feature_names[col]} by Class")
    st.pyplot(plt)

# ------------------------------
# Step 7: Predict New Sample
# ------------------------------
st.subheader("🖊️ Predict a New Drawing")
st.write("Enter your new sample measurements below:")

new_data = {}
for col in feature_cols:
    new_data[col] = st.number_input(feature_names[col], value=float(df[col].mean()))

if st.button("Predict"):
    new_sample = pd.DataFrame([new_data])
    prediction = clf.predict(new_sample)[0]
    probability = clf.predict_proba(new_sample)[0]
    class_map = {0: "Healthy", 1: "Parkinson's"}
    st.success(f"Predicted class: {class_map[prediction]}")
    st.info(f"Probability - Healthy: {probability[0]:.2f}, Parkinson's: {probability[1]:.2f}")

# ------------------------------
# Step 8: Save Model (optional)
# ------------------------------
joblib.dump(clf, "handpd_model.pkl")
