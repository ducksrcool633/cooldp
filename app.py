# ==============================
# Kaggle Notebook: Parkinson's Spiral HandPD Model (Cleaned & Warning-Free)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ==============================
# Step 0: Load Dataset
# ==============================
spiral_df = pd.read_csv("/kaggle/input/spiral-handpd/Spiral_HandPD.csv")

# Map CLASS_TYPE: 0 = Healthy, 1 = Parkinson's
spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1

# Define numeric features
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
# Step 1: Data Exploration
# ==============================
print("=== First 5 Rows ===")
print(spiral_df.head())

print("\n=== Column Data Types ===")
print(spiral_df.dtypes)

print("\n=== Missing Values ===")
print(spiral_df.isnull().sum())

print("\n=== Numeric Feature Summary ===")
print(spiral_df[numeric_features].describe())

print("\n=== Class Distribution ===")
print(spiral_df['CLASS_TYPE'].value_counts())

# ==============================
# Step 2: Feature Visualization
# ==============================
# Boxplots with mean diamonds
num_features = len(numeric_features)
cols = 3
rows = int(np.ceil(num_features / cols))

plt.figure(figsize=(18, 5*rows))
for i, feature in enumerate(numeric_features):
    plt.subplot(rows, cols, i+1)
    ax = sns.boxplot(x='CLASS_TYPE', y=feature, data=spiral_df, showfliers=True)
    
    # Plot mean as red diamond
    means = spiral_df.groupby('CLASS_TYPE')[feature].mean().values
    for cls, mean_val in enumerate(means):
        ax.scatter(cls, mean_val, color='red', marker='D', s=50, label='Mean' if cls==0 else "")
    
    plt.xticks([0,1], ['Healthy', "Parkinson's"])
    plt.title(f"{feature} by Class")
    plt.xlabel("Class")
    plt.ylabel(feature)
    
    if i==0:
        plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Scatterplot RMS vs MRT
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='RMS', 
    y='MRT', 
    hue='CLASS_TYPE', 
    data=spiral_df, 
    palette=['green','orange'], 
    s=70
)
plt.xlabel("RMS (magnitude of pen movement)")
plt.ylabel("MRT (mean relative tremor)")
plt.title("RMS vs MRT by Class")
plt.legend(labels=['Healthy','Parkinson\'s'])
plt.show()

# ==============================
# Step 3: Random Forest Modeling
# ==============================
X = spiral_df[numeric_features].values
y = spiral_df['CLASS_TYPE'].values
groups = spiral_df['ID_PATIENT'].values

gkf = GroupKFold(n_splits=5)

print("\n=== GroupKFold Cross-Validation ===")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n--- Fold {fold+1} ---")
    print(f"F1 Score: {f1:.3f}, ROC AUC: {auc:.3f}")
    print("Confusion Matrix:")
    print(cm)

# ==============================
# Step 4: Feature Importance
# ==============================
importances = clf.feature_importances_
feat_importance_df = pd.DataFrame({'feature': numeric_features, 'importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=feat_importance_df)
plt.title("Feature Importance - Random Forest")
plt.show()

print("\n✅ Notebook complete! Dashboard, modeling, and feature importance ready.")

# ==============================
# Parkinson's Prediction - Full Plug-and-Play Block
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# --- Step 1: Load dataset ---
spiral_df = pd.read_csv("/kaggle/input/spiral-handpd/Spiral_HandPD.csv")

# Map CLASS_TYPE: 0 = Healthy, 1 = Parkinson's
spiral_df['CLASS_TYPE'] = spiral_df['CLASS_TYPE'] - 1

# --- Step 2: Define features and labels ---
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

X = spiral_df[feature_cols]
y = spiral_df['CLASS_TYPE']

# --- Step 3: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 4: Train Random Forest model ---
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# --- Step 5: Evaluate model ---
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]

print("=== Model Performance on Test Set ===")
print("F1 Score:", round(f1_score(y_test, preds), 3))
print("ROC AUC:", round(roc_auc_score(y_test, probs), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# --- Step 6: Predict new sample(s) ---
# Replace the example values below with your new measurements
new_sample = pd.DataFrame({
    'RMS': [1200],  
    'MAX_BETWEEN_ET_HT': [30],
    'MIN_BETWEEN_ET_HT': [5],
    'STD_DEVIATION_ET_HT': [12],
    'MRT': [0.8],
    'MAX_HT': [150],
    'MIN_HT': [20],
    'STD_HT': [25],
    'CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT': [10]
})

# Predict class and probability
prediction = clf.predict(new_sample)
probability = clf.predict_proba(new_sample)

class_map = {0: "Healthy", 1: "Parkinson's"}
print("\n=== Prediction for New Sample ===")
print("Predicted class:", class_map[prediction[0]])
print("Probability:", f"Healthy={probability[0][0]:.2f}, Parkinson's={probability[0][1]:.2f}")

# ==============================
# ✅ Done. Just replace values in new_sample to predict new drawings.
import joblib
joblib.dump(clf, "handpd_model.pkl")
