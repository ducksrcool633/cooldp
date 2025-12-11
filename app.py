# ==============================
# Streamlit App: Parkinson's HandPD Predictor
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

st.set_page_config(page_title="HandPD Predictor", layout="wide")

st.title("🖊️ Parkinson's Hand Tremor Predictor")
st.write("Upload a Spiral HandPD CSV to explore, train, and predict Parkinson's.")

# -----------------------------
# Step 1: CSV Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your Spiral HandPD CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ CSV loaded successfully!")
        st.write("Columns detected:", df.columns.tolist())

        # -----------------------------
        # Step 2: Fix column names
        # -----------------------------
        expected_cols = [
            'ID_PATIENT',
            'CLASS_TYPE',
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

        if len(df.columns) < 11:
            st.error("CSV does not have enough columns!")
        else:
            df = df.iloc[:, :11]  # take first 11 columns
            df.columns = expected_cols
            st.write("✅ Columns renamed to standard names.")

        # Map CLASS_TYPE
        df['CLASS_TYPE'] = df['CLASS_TYPE'].astype(int) - 1  # 0=Healthy, 1=Parkinson's

        # -----------------------------
        # Step 3: Explore data
        # -----------------------------
        st.subheader("📊 Data Preview")
        st.write(df.head())

        st.subheader("📈 Feature Distributions")
        numeric_features = expected_cols[2:]  # all numeric columns
        for feature in numeric_features:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x='CLASS_TYPE', y=feature, data=df, showfliers=True, ax=ax)
            means = df.groupby('CLASS_TYPE')[feature].mean().values
            for cls, mean_val in enumerate(means):
                ax.scatter(cls, mean_val, color='red', marker='D', s=50)
            ax.set_xticklabels(['Healthy','Parkinson\'s'])
            ax.set_title(f"{feature} by Class (Red diamond = mean)")
            st.pyplot(fig)

        # Scatterplot RMS vs MRT
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x='RMS', y='MRT', hue='CLASS_TYPE', data=df,
                        palette=['green','orange'], s=70, ax=ax)
        ax.set_xlabel("RMS (pen movement magnitude)")
        ax.set_ylabel("MRT (mean relative tremor)")
        ax.set_title("RMS vs MRT by Class")
        ax.legend(['Healthy','Parkinson\'s'])
        st.pyplot(fig)

        # -----------------------------
        # Step 4: Train model
        # -----------------------------
        st.subheader("🤖 Training Random Forest Model...")
        X = df[numeric_features]
        y = df['CLASS_TYPE']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:,1]
        st.write("**F1 Score:**", round(f1_score(y_test, preds),3))
        st.write("**ROC AUC:**", round(roc_auc_score(y_test, probs),3))
        st.write("**Confusion Matrix:**")
        st.write(confusion_matrix(y_test, preds))

        # Feature importance
        importances = clf.feature_importances_
        feat_imp_df = pd.DataFrame({'feature': numeric_features, 'importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x='importance', y='feature', data=feat_imp_df, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # Save model
        joblib.dump(clf, "handpd_model.pkl")
        st.success("✅ Model trained and saved!")

        # -----------------------------
        # Step 5: Predict new sample
        # -----------------------------
        st.subheader("📝 Predict New Sample")
        st.write("Enter values to predict Parkinson's:")

        user_input = {}
        for feature in numeric_features:
            user_input[feature] = st.number_input(feature, value=float(df[feature].mean()))

        if st.button("Predict"):
            new_sample = pd.DataFrame([user_input])
            prediction = clf.predict(new_sample)[0]
            probability = clf.predict_proba(new_sample)[0]

            class_map = {0:"Healthy", 1:"Parkinson's"}
            st.write(f"**Predicted class:** {class_map[prediction]}")
            st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

    except Exception as e:
        st.error(f"Error reading CSV or processing data: {e}")

else:
    st.info("Please upload a CSV file to continue.")
