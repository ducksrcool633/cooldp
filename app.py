import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.title("🖊️ Parkinson's Spiral Hand Detection")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload your Spiral HandPD CSV", type=["csv"])
if uploaded_file is not None:
    try:
        # Load CSV, ignore extra columns
        spiral_df = pd.read_csv(uploaded_file)
        
        # Keep only the columns we need
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

        missing_cols = [col for col in expected_cols if col not in spiral_df.columns]
        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
            st.stop()

        spiral_df = spiral_df[expected_cols]  # keep only expected columns

        st.success("CSV loaded successfully!")
        st.write("First 5 rows:")
        st.dataframe(spiral_df.head())

        # --- Step 2: Feature Visualization ---
        numeric_features = expected_cols[2:]  # all features except ID and CLASS_TYPE

        st.subheader("📊 Feature Distributions by Class")
        for feature in numeric_features:
            plt.figure(figsize=(5,3))
            sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True)
            means = spiral_df.groupby('CLASS_TYPE')[feature].mean().values
            for cls, mean_val in enumerate(means):
                plt.scatter(cls, mean_val, color='red', marker='D', s=40)
            plt.xticks([0,1], ['Healthy','Parkinson\'s'])
            plt.title(f"{feature} by Class")
            plt.xlabel("Class")
            plt.ylabel(feature)
            st.pyplot(plt)
            plt.clf()

        # --- Step 3: Model Training ---
        X = spiral_df[numeric_features]
        y = spiral_df['CLASS_TYPE']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:,1]

        st.subheader("🎯 Model Performance")
        st.write(f"F1 Score: {f1_score(y_test, preds):.3f}")
        st.write(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, preds))

        # --- Step 4: Predict New Sample ---
        st.subheader("📝 Predict New Sample")
        st.write("Enter values for a new drawing:")

        new_data = {}
        for feature in numeric_features:
            new_data[feature] = [st.number_input(feature, value=float(spiral_df[feature].mean()))]

        new_sample = pd.DataFrame(new_data)
        prediction = clf.predict(new_sample)[0]
        probability = clf.predict_proba(new_sample)[0]

        class_map = {0: "Healthy", 1: "Parkinson's"}
        st.write(f"**Predicted class:** {class_map[prediction]}")
        st.write(f"**Probability:** Healthy={probability[0]:.2f}, Parkinson's={probability[1]:.2f}")

        # Optional: save model
        joblib.dump(clf, "handpd_model.pkl")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("Please upload your CSV to start!")
