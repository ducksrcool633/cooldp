
# ==============================
# Streamlit App: Parkinson's Spiral HandPD
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

st.title("Parkinson's Handwriting Detection")

# ==============================
# 1. Load CSV safely
# ==============================
uploaded_file = st.file_uploader("Upload Spiral HandPD CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.ParserError:
        df = pd.read_csv(uploaded_file, header=None)
        df.columns = [
            'ID_PATIENT', 'CLASS_TYPE', 'RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
            'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT', 'STD_HT',
            'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT'
        ]

    # Fix extra unnamed columns if any
    df = df.iloc[:, :11]

    # ==============================
    # 2. Rename features for clarity
    # ==============================
    rename_map = {
        'RMS': 'Pen Movement Size',
        'MAX_BETWEEN_ET_HT': 'Max Time Between Dots',
        'MIN_BETWEEN_ET_HT': 'Min Time Between Dots',
        'STD_DEVIATION_ET_HT': 'Time Variability',
        'MRT': 'Average Tremor',
        'MAX_HT': 'Max Height',
        'MIN_HT': 'Min Height',
        'STD_HT': 'Height Variability',
        'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT': 'Dot Direction Changes'
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure CLASS_TYPE is numeric and mapped
    if df['CLASS_TYPE'].dtype != np.number:
        df['CLASS_TYPE'] = pd.to_numeric(df['CLASS_TYPE'], errors='coerce')
    df.dropna(subset=['CLASS_TYPE'], inplace=True)
    df['CLASS_TYPE'] = df['CLASS_TYPE'].astype(int) - 1  # 0=Healthy, 1=Parkinson's

    # ==============================
    # 3. Display basic info
    # ==============================
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write("### Class Distribution")
    st.bar_chart(df['CLASS_TYPE'].value_counts())

    # ==============================
    # 4. Visualizations
    # ==============================
    st.write("### Feature Boxplots")
    numeric_features = list(rename_map.values())
    num_features = len(numeric_features)
    cols = 3
    rows = int(np.ceil(num_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = axes.flatten()
    for i, feature in enumerate(numeric_features):
        sns.boxplot(x='CLASS_TYPE', y=feature, data=df, ax=axes[i])
        means = df.groupby('CLASS_TYPE')[feature].mean().values
        for cls, mean_val in enumerate(means):
            axes[i].scatter(cls, mean_val, color='red', marker='D', s=40)
        axes[i].set_xticklabels(['Healthy', "Parkinson's"])
        axes[i].set_title(feature)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    # Scatterplot example
    st.write("### Pen Movement vs Average Tremor")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x='Pen Movement Size',
        y='Average Tremor',
        hue='CLASS_TYPE',
        data=df,
        palette=['green','orange'],
        s=50,
        ax=ax2
    )
    ax2.set_xlabel("Pen Movement Size")
    ax2.set_ylabel("Average Tremor")
    ax2.set_title("Pen Movement vs Tremor")
    ax2.legend(labels=['Healthy','Parkinson\'s'])
    st.pyplot(fig2)

    # ==============================
    # 5. Train Model
    # ==============================
    st.write("### Training Random Forest Model...")
    feature_cols = numeric_features
    X = df[feature_cols]
    y = df['CLASS_TYPE']

    # Check if enough data to stratify
    if len(df['CLASS_TYPE'].value_counts()) < 2:
        st.error("Not enough data to train the model.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:,1]

        st.write("**Model Performance on Test Set:**")
        st.write(f"F1 Score: {round(f1_score(y_test, preds),3)}")
        st.write(f"ROC AUC: {round(roc_auc_score(y_test, probs),3)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, preds))

        # Save model
        joblib.dump(clf, "handpd_model.pkl")
        st.success("Model trained and saved successfully!")

        # ==============================
        # 6. Predict new sample
        # ==============================
        st.write("### Predict New Sample")
        new_input = {}
        for feature in feature_cols:
            val = st.number_input(f"Enter {feature}", value=0.0)
            new_input[feature] = [val]

        if st.button("Predict"):
            new_df = pd.DataFrame(new_input)
            prediction = clf.predict(new_df)[0]
            probability = clf.predict_proba(new_df)[0]
            class_map = {0:"Healthy",1:"Parkinson's"}
            st.write(f"**Predicted class:** {class_map[prediction]}")
            st.write(f"Probability: Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")
