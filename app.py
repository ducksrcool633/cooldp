# ==============================
# HandPD Streamlit App
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

st.set_page_config(page_title="HandPD Spiral Dashboard", layout="wide")

st.title("🖐 HandPD Spiral Hand Analysis")

# ==============================
# Load dataset
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Spiral_HandPD.csv")  # CSV must be in same folder as app.py
        df.columns = [c.strip() for c in df.columns]  # remove extra spaces
        if 'CLASS_TYPE' not in df.columns:
            st.error("Error: 'CLASS_TYPE' column not found in CSV!")
            st.stop()
        df['CLASS_TYPE'] = df['CLASS_TYPE'] - 1  # 0=Healthy, 1=Parkinson's
        return df
    except FileNotFoundError:
        st.error("CSV file not found! Upload Spiral_HandPD.csv in the app folder.")
        st.stop()

spiral_df = load_data()

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Rows: {spiral_df.shape[0]}, Columns: {spiral_df.shape[1]}")
st.sidebar.write("Class Distribution:")
st.sidebar.write(spiral_df['CLASS_TYPE'].value_counts().rename({0:"Healthy", 1:"Parkinson's"}))

# ==============================
# Numeric features
# ==============================
numeric_features = [
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

# ==============================
# Dashboard: Boxplots
# ==============================
st.subheader("📊 Feature Distributions by Class")

cols = 3
rows = int(np.ceil(len(numeric_features)/cols))
fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    ax = axes[i]
    sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True, ax=ax)
    means = spiral_df.groupby('CLASS_TYPE')[feature].mean().values
    for cls, mean_val in enumerate(means):
        ax.scatter(cls, mean_val, color='red', marker='D', s=50)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Healthy', "Parkinson's"])
    ax.set_title(feature)
    ax.set_xlabel("Class")
    ax.set_ylabel("Value")
    if i==0:
        ax.legend(["Mean (red diamond)"])
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

st.pyplot(fig)

# ==============================
# Scatterplot RMS vs MRT
# ==============================
st.subheader("✏ RMS vs MRT")
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.scatterplot(
    x='RMS',
    y='MRT',
    hue='CLASS_TYPE',
    data=spiral_df,
    palette=['green','orange'],
    s=70,
    ax=ax2
)
ax2.set_xlabel("RMS (magnitude of pen movement)")
ax2.set_ylabel("MRT (mean relative tremor)")
ax2.set_title("RMS vs MRT by Class")
ax2.legend(title="Class", labels=['Healthy', "Parkinson's"])
st.pyplot(fig2)

# ==============================
# Train Random Forest Model
# ==============================
st.subheader("🤖 Train Model & Predict")

X = spiral_df[numeric_features]
y = spiral_df['CLASS_TYPE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

st.write("### Model Performance on Test Set")
st.write(f"F1 Score: {round(f1_score(y_test, preds),3)}")
st.write(f"ROC AUC: {round(roc_auc_score(y_test, probs),3)}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, preds))

# ==============================
# Predict New Sample
# ==============================
st.subheader("🧪 Predict New Sample")

st.write("Enter measurements for a new drawing:")

new_sample_dict = {}
for feature in numeric_features:
    val = st.number_input(feature, value=float(spiral_df[feature].mean()))
    new_sample_dict[feature] = [val]

new_sample_df = pd.DataFrame(new_sample_dict)

if st.button("Predict"):
    prediction = clf.predict(new_sample_df)[0]
    probability = clf.predict_proba(new_sample_df)[0]
    class_map = {0:"Healthy", 1:"Parkinson's"}
    st.write(f"**Predicted class:** {class_map[prediction]}")
    st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

# ==============================
# Save model (optional)
# ==============================
joblib.dump(clf, "handpd_model.pkl")
st.success("✅ Model trained and saved as handpd_model.pkl")

