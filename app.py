# ==============================
# HandPD Streamlit App - Fully Updated
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

st.set_page_config(page_title="HandPD Parkinson's Prediction", layout="wide")

st.title("HandPD Parkinson's Prediction & Dashboard")
st.markdown("""
This app allows you to explore the HandPD dataset, visualize tremor features, 
and predict Parkinson's disease from new samples.
""")

# ==============================
# Upload CSV
# ==============================
st.sidebar.header("Upload your Spiral_HandPD CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    try:
        spiral_df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    st.info("Please upload a Spiral_HandPD CSV to continue.")
    st.stop()

# Check for CLASS_TYPE column
if 'CLASS_TYPE' not in spiral_df.columns:
    st.error("CSV does not contain 'CLASS_TYPE' column!")
    st.write("Columns detected:", spiral_df.columns.tolist())
    st.stop()

# Map CLASS_TYPE: 0 = Healthy, 1 = Parkinson's
spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1

# ==============================
# Numeric Features
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

# Ensure all numeric features exist
missing_features = [f for f in numeric_features if f not in spiral_df.columns]
if missing_features:
    st.error(f"Missing features in CSV: {missing_features}")
    st.stop()

# ==============================
# Data Exploration
# ==============================
st.header("Dataset Overview")
st.write("First 5 rows of your dataset:")
st.dataframe(spiral_df.head())

st.write("Class distribution:")
st.bar_chart(spiral_df['CLASS_TYPE'].value_counts().rename({0:'Healthy',1:"Parkinson's"}))

# ==============================
# Feature Dashboard
# ==============================
st.header("Feature Visualizations")

num_features = len(numeric_features)
cols = 3
rows = int(np.ceil(num_features / cols))

fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, ax=axes[i], showfliers=True)
    # plot means
    means = spiral_df.groupby('CLASS_TYPE')[feature].mean().values
    for cls, mean_val in enumerate(means):
        axes[i].scatter(cls, mean_val, color='red', marker='D', s=50)
    axes[i].set_xticklabels(['Healthy', "Parkinson's"])
    axes[i].set_title(feature)
    axes[i].set_xlabel("Class")
    axes[i].set_ylabel("Value")

# Remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

st.pyplot(fig)

# RMS vs MRT scatterplot
st.subheader("RMS vs MRT Scatterplot")
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.scatterplot(x='RMS', y='MRT', hue='CLASS_TYPE', data=spiral_df,
                palette=['green','orange'], s=70, ax=ax2)
ax2.set_xlabel("RMS (magnitude of pen movement)")
ax2.set_ylabel("MRT (mean relative tremor)")
ax2.set_title("RMS vs MRT by Class")
ax2.legend(labels=['Healthy','Parkinson\'s'])
st.pyplot(fig2)

# ==============================
# Train Random Forest Model
# ==============================
st.header("Train Random Forest Model")
test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5)/100

X = spiral_df[numeric_features]
y = spiral_df['CLASS_TYPE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

st.subheader("Model Performance on Test Set")
st.write(f"F1 Score: {f1_score(y_test, preds):.3f}")
st.write(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, preds))

# Feature importance
feat_importance_df = pd.DataFrame({'feature': numeric_features, 'importance': clf.feature_importances_})
feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)
st.subheader("Feature Importance")
st.bar_chart(feat_importance_df.set_index('feature'))

# Save model
joblib.dump(clf, "handpd_model.pkl")
st.success("Random Forest model trained and saved!")

# ==============================
# Predict New Sample
# ==============================
st.header("Predict New Sample")
st.write("Enter numeric values for a new drawing:")

new_sample_dict = {}
for feature in numeric_features:
    new_sample_dict[feature] = st.number_input(feature, value=float(spiral_df[feature].mean()))

new_sample_df = pd.DataFrame(new_sample_dict, index=[0])

if st.button("Predict"):
    pred_class = clf.predict(new_sample_df)[0]
    pred_prob = clf.predict_proba(new_sample_df)[0]
    st.write(f"Predicted class: {'Healthy' if pred_class==0 else 'Parkinson\'s'}")
    st.write(f"Probability: Healthy={pred_prob[0]:.2f}, Parkinson's={pred_prob[1]:.2f}")



