import cv2
import os
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================
# Dataset folders
# ==============================

train_healthy = "dataset/AGM_HS/healthy"
train_stressed = "dataset/AGM_HS/stressed"

test_healthy = "dataset/AGM_HS/test/healthy"
test_stressed = "dataset/AGM_HS/test/stressed"


# ==============================
# Feature extraction
# ==============================

def extract_features(image):

    image = cv2.resize(image, (128, 128))

    # HSV color features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    color_features = np.concatenate([
        hist_h,
        hist_s,
        hist_v
    ])

    # HOG features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Edge features
    edges = cv2.Canny(gray, 50, 150)

    edge_features = cv2.resize(
        edges, (32, 32)
    ).flatten() / 255.0

    # Combine features
    return np.concatenate([
        color_features,
        hog_features,
        edge_features
    ])


# ==============================
# Load training data
# ==============================

X_train = []
y_train = []

print("Loading training images...")


for file in os.listdir(train_healthy):

    path = os.path.join(train_healthy, file)
    image = cv2.imread(path)

    if image is not None:
        X_train.append(extract_features(image))
        y_train.append(0)


for file in os.listdir(train_stressed):

    path = os.path.join(train_stressed, file)
    image = cv2.imread(path)

    if image is not None:
        X_train.append(extract_features(image))
        y_train.append(1)


X_train = np.array(X_train)
y_train = np.array(y_train)

print("Training images loaded:", len(X_train))
print("Feature count:", X_train.shape[1])


# ==============================
# Random Forest model
# ==============================

print("Training Random Forest AI model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# ==============================
# Load test data
# ==============================

X_test = []
y_test = []

print("Loading test images...")


for file in os.listdir(test_healthy):

    path = os.path.join(test_healthy, file)
    image = cv2.imread(path)

    if image is not None:
        X_test.append(extract_features(image))
        y_test.append(0)


for file in os.listdir(test_stressed):

    path = os.path.join(test_stressed, file)
    image = cv2.imread(path)

    if image is not None:
        X_test.append(extract_features(image))
        y_test.append(1)


X_test = np.array(X_test)
y_test = np.array(y_test)

print("Test images loaded:", len(X_test))


# ==============================
# Evaluate model
# ==============================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


print()
print("===================================")
print("RANDOM FOREST MODEL PERFORMANCE")
print("===================================")

print("Test Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Healthy", "Stressed"],
        zero_division=0
    )
)


# ==============================
# Save model
# ==============================

joblib.dump(model, "crop_health_model.pkl")

print()
print("Random Forest model saved successfully!")
print("File: crop_health_model.pkl")