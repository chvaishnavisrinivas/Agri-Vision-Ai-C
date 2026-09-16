import cv2
import os
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load trained model
model = joblib.load("crop_health_model.pkl")


test_healthy_folder = "dataset/test/healthy"
test_stressed_folder = "dataset/test/stressed"


X_test = []
y_test = []


def extract_features(image):
    image = cv2.resize(image, (128, 128))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edge_features = cv2.resize(
        edges,
        (32, 32)
    ).flatten() / 255.0

    return np.concatenate([
        hist_h,
        hist_s,
        hist_v,
        edge_features
    ])


# Load healthy test images
for file in os.listdir(test_healthy_folder):

    path = os.path.join(test_healthy_folder, file)

    image = cv2.imread(path)

    if image is not None:
        X_test.append(extract_features(image))
        y_test.append(0)


# Load stressed test images
for file in os.listdir(test_stressed_folder):

    path = os.path.join(test_stressed_folder, file)

    image = cv2.imread(path)

    if image is not None:
        X_test.append(extract_features(image))
        y_test.append(1)


X_test = np.array(X_test)
y_test = np.array(y_test)


print("Test images loaded:", len(X_test))


# Predict test images
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")


# Classification report
print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Healthy", "Stressed"],
    zero_division=0
))


# Confusion matrix
print("Confusion Matrix:")

print(confusion_matrix(y_test, predictions))