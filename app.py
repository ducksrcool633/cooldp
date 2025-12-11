import streamlit as st
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from skimage import io, color

st.title("Parkinson's Spiral Hand Prediction (Image Input)")

# -------------------------------
# Step 1: Upload spiral hand image
# -------------------------------
uploaded_image = st.file_uploader("Upload Spiral Hand Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Read image as grayscale
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded Spiral Hand", use_column_width=True)

    # -------------------------------
    # Step 2: Convert image to features
    # -------------------------------
    # Resize to fixed size
    img_resized = cv2.resize(img, (200, 200))

    # Flatten
    flat = img_resized.flatten()
    
    # Compute features
    features = {
        'RMS': np.sqrt(np.mean(flat**2)),
        'MAX_ET': np.max(flat),
        'MIN_ET': np.min(flat),
        'STD_ET': np.std(flat),
        'MRT': np.mean(flat),
        'MAX_HT': np.max(np.mean(img_resized, axis=0)),  # max of column means
        'MIN_HT': np.min(np.mean(img_resized, axis=0)),
        'STD_HT': np.std(np.mean(img_resized, axis=0)),
        'CHANGES_ET': np.sum(np.diff((flat>128).astype(int)) != 0)
    }

    st.write("Extracted Features:")
    st.json(features)

    # -------------------------------
    # Step 3: Train dummy model
    # -------------------------------
    # For demonstration, create random dataset
    np.random.seed(42)
    X_dummy = np.random.rand(100, 9)
    y_dummy = np.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # -------------------------------
    # Step 4: Predict
    # -------------------------------
    sample = np.array(list(features.values())).reshape(1, -1)
    pred_class = clf.predict(sample)[0]
    pred_proba = clf.predict_proba(sample)[0]

    class_map = {0: "Healthy", 1: "Parkinson's"}
    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {class_map[pred_class]}")
    st.write(f"**Probability:** Healthy={pred_proba[0]:.2f}, Parkinson's={pred_proba[1]:.2f}")

else:
    st.info("Upload a spiral hand image to predict.")
