# ==============================
# Streamlit App: HandPD Parkinson's Prediction
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("HandPD Parkinson's Prediction Dashboard")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload Spiral_HandPD CSV", type="csv")

if uploaded_file is not None:
    # Try reading CSV with header
    try:
        spiral_df = pd.read_csv(uploaded_file, header=0)
        if 'CLASS_TYPE' not in spiral_df.columns:
            raise ValueError("CLASS_TYPE not found")
    except:
        # If header missing or wrong, manually assign columns
        column_names = [
            'ID_EXAM', 'IMAGE_NAME', 'ID_PATIENT', 'CLASS_TYPE', 'GENDER',
            'RIGH/LEFT-HANDED', 'AGE', 'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
            'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]
        spiral_df = pd.read_csv(uploaded_file, header=None, names=column_names)
    
    st.success("CSV Loaded Successfully!")
    st.write("Columns detected:", list(spiral_df.columns))

    # Map CLASS_TYPE: 0 = Healthy, 1 = Parkinson's
    spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1

    # --- Numeric features ---
    feature_cols = [
        'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]

    # --- Step 2: Data Exploration ---
    st.subheader("Class Distribution")
    st.bar_chart(spiral_df['CLASS_TYPE'].value_counts().rename({0:'Healthy',1:"Parkinson's"}))

    # --- Step 3: Boxplot Dashboard ---
    st.subheader("Feature Boxplots")
    num_features = len(feature_cols)
    cols = 3
    rows = int(np.ceil(num_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True, ax=ax)
        means = spiral_df.groupby('CLASS_TYPE')[feature].mean().values
        for cls, mean_val in enumerate(means):
            ax.scatter(cls, mean_val, color='red', marker='D', s=50, label='Mean' if cls==0 else "")
        ax.set_xticklabels(['Healthy', "Parkinson's"])
        ax.set_title(feature)
        if i==0:
            ax.legend(loc='upper right')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    st.pyplot(fig)

    # --- Step 4: Train Random Forest ---
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    st.subheader("Model Performance")
    st.write("F1 Score:", round(f1_score(y_test, preds),3))
    st.write("ROC AUC:", round(roc_auc_score(y_test, probs),3))
    st.write("Confusion Matrix:", confusion_matrix(y_test, preds))
    
    # --- Step 5: Predict new sample ---
    st.subheader("Predict New Sample")
    new_data = {}
    for feature in feature_cols:
        val = st.number_input(f"{feature}:", value=float(0))
        new_data[feature] = [val]
    
    if st.button("Predict"):
        new_sample = pd.DataFrame(new_data)
        prediction = clf.predict(new_sample)[0]
        probability = clf.predict_proba(new_sample)[0]
        class_map = {0: "Healthy", 1: "Parkinson's"}
        st.write("Predicted Class:", class_map[prediction])
        st.write(f"Probability: Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")


