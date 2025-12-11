# ==============================
# Streamlit App: Parkinson's Spiral HandPD Model
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import joblib

st.title("Parkinson's Hand Tremor Predictor")

# ------------------------------
# Step 0: Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Upload your Spiral_HandPD CSV", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload a CSV to continue.")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded_file, header=0)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# ------------------------------
# Step 1: Ensure correct columns
# ------------------------------
expected_cols = [
    'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
    'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
    'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
]

# Assign columns if missing headers
if set(expected_cols).issubset(df.columns):
    st.success("All expected columns found!")
else:
    if df.shape[1] >= len(expected_cols):
        df = df.iloc[:, :len(expected_cols)]
        df.columns = expected_cols
        st.info("Columns assigned manually.")
    else:
        st.error(f"CSV missing required columns! Found: {list(df.columns)}")
        st.stop()

# Convert CLASS_TYPE to numeric
if df['CLASS_TYPE'].dtype not in [np.int64, np.float64]:
    df['CLASS_TYPE'] = pd.to_numeric(df['CLASS_TYPE'], errors='coerce')
df.dropna(subset=['CLASS_TYPE'], inplace=True)
df['CLASS_TYPE'] = df['CLASS_TYPE'].astype(int) - 1  # 0=Healthy, 1=Parkinson's

# ------------------------------
# Step 2: Data Exploration
# ------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Class Distribution")
st.bar_chart(df['CLASS_TYPE'].value_counts())

# ------------------------------
# Step 3: Feature Visualization
# ------------------------------
numeric_features = [
    'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT', 'STD_DEVIATION_ET_HT',
    'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT', 'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
]

st.subheader("Feature Distributions by Class")
cols = 3
rows = int(np.ceil(len(numeric_features) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))

for i, feature in enumerate(numeric_features):
    r = i // cols
    c = i % cols
    sns.boxplot(
        x='CLASS_TYPE', y=feature, data=df, ax=axes[r, c],
        showfliers=True, palette=['green', 'orange']
    )
    axes[r, c].set_xticklabels(['Healthy','Parkinson\'s'])
    axes[r, c].set_ylabel(feature)
    axes[r, c].set_xlabel("Class")
    axes[r, c].set_title(feature)

plt.tight_layout()
st.pyplot(fig)

# ------------------------------
# Step 4: Train Random Forest
# ------------------------------
X = df[numeric_features]
y = df['CLASS_TYPE']

# Check that there is more than 1 sample per class
if min(df['CLASS_TYPE'].value_counts()) < 2:
    st.error("Not enough samples per class to train model. Each class needs at least 2 samples.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

st.subheader("Model Performance")
st.write(f"F1 Score: {f1_score(y_test, preds):.3f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, preds))

# Feature Importance
feat_importance_df = pd.DataFrame({'feature': numeric_features, 'importance': clf.feature_importances_})
feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)

st.subheader("Feature Importance")
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=feat_importance_df, ax=ax2)
ax2.set_xlabel("Importance")
ax2.set_ylabel("Feature")
plt.tight_layout()
st.pyplot(fig2)

# Save model
joblib.dump(clf, "handpd_model.pkl")

# ------------------------------
# Step 5: Predict new sample
# ------------------------------
st.subheader("Predict New Sample")

def user_input_features():
    inputs = {}
    for feature in numeric_features:
        inputs[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
    return pd.DataFrame(inputs, index=[0])

new_sample = user_input_features()

if st.button("Predict"):
    prediction = clf.predict(new_sample)[0]
    probability = clf.predict_proba(new_sample)[0]
    class_map = {0: "Healthy", 1: "Parkinson's"}
    st.write(f"**Predicted class:** {class_map[prediction]}")
    st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")
