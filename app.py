# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import joblib

st.title("Parkinson's Hand Tremor Prediction Dashboard")

# --- Step 1: Load CSV ---
uploaded_file = st.file_uploader("Upload Spiral_HandPD CSV", type="csv")

if uploaded_file is not None:
    spiral_df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully!")
    
    # Check columns
    expected_cols = [
        'ID_PATIENT','CLASS_TYPE','RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT','MRT','MAX_HT','MIN_HT','STD_HT',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    if all(col in spiral_df.columns for col in expected_cols):
        st.write("Columns detected:", list(spiral_df.columns))
    else:
        st.error(f"CSV missing expected columns! Found: {list(spiral_df.columns)}")
        st.stop()

    # Map CLASS_TYPE
    spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1  # 0=Healthy,1=Parkinson's
    
    # Define numeric features
    feature_cols = [
        'RMS','MAX_BETWEEN_ET_HT','MIN_BETWEEN_ET_HT',
        'STD_DEVIATION_ET_HT','MRT','MAX_HT','MIN_HT','STD_HT',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
    ]
    
    # --- Step 2: Dashboard ---
    st.subheader("Feature Distributions")
    num_features = len(feature_cols)
    cols = 3
    rows = int(np.ceil(num_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        sns.boxplot(
            x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True, ax=axes[i]
        )
        means = spiral_df.groupby('CLASS_TYPE')[feature].mean().values
        for cls, mean_val in enumerate(means):
            axes[i].scatter(cls, mean_val, color='red', marker='D', s=50, label='Mean' if cls==0 else "")
        axes[i].set_xticklabels(['Healthy','Parkinson\'s'])
        axes[i].set_title(feature)
        axes[i].set_xlabel("Class")
        axes[i].set_ylabel("Value")
        if i==0:
            axes[i].legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Scatter RMS vs MRT
    st.subheader("RMS vs MRT Scatterplot")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.scatterplot(
        x='RMS', y='MRT', hue='CLASS_TYPE', data=spiral_df,
        palette=['green','orange'], s=70, ax=ax2
    )
    ax2.set_xlabel("RMS (magnitude of pen movement)")
    ax2.set_ylabel("MRT (mean relative tremor)")
    ax2.set_title("RMS vs MRT by Class")
    ax2.legend(labels=['Healthy','Parkinson\'s'])
    st.pyplot(fig2)
    
    # --- Step 3: Model Training ---
    X = spiral_df[feature_cols]
    y = spiral_df['CLASS_TYPE']
    
    class_counts = y.value_counts()
    st.write("Class counts:", class_counts.to_dict())
    
    if (class_counts >= 2).all():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        st.warning("Not enough samples for stratified split. Using normal split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    st.subheader("Model Evaluation on Test Set")
    st.write("F1 Score:", round(f1_score(y_test, preds),3))
    st.write("ROC AUC:", round(roc_auc_score(y_test, probs),3))
    st.write("Confusion Matrix:", confusion_matrix(y_test, preds))
    
    # Save model
    joblib.dump(clf, "handpd_model.pkl")
    
    # --- Step 4: Predict New Sample ---
    st.subheader("Predict a New Sample")
    new_sample_data = {}
    for feature in feature_cols:
        new_sample_data[feature] = [st.number_input(f"{feature}", value=float(0))]
    
    if st.button("Predict"):
        new_sample = pd.DataFrame(new_sample_data)
        pred = clf.predict(new_sample)[0]
        prob = clf.predict_proba(new_sample)[0]
        class_map = {0:"Healthy",1:"Parkinson's"}
        st.write("Predicted class:", class_map[pred])
        st.write(f"Probability: Healthy={prob[0]:.2f}, Parkinson's={prob[1]:.2f}")
else:
    st.info("Please upload a CSV to continue.")
