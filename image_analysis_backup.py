import cv2
import numpy as np
import joblib
from skimage.feature import hog

model = joblib.load("crop_health_model.pkl")


# --------------------------------------------------
# 1. Detect the main crop/leaf region
# --------------------------------------------------
def detect_crop_region(image):
    image = cv2.resize(image, (512, 512))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green vegetation mask
    lower_green = np.array([25, 30, 20])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image, mask

    # Select largest vegetation region
    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)
    image_area = image.shape[0] * image.shape[1]

    # If detected region is too small, use complete image
    if area < 0.05 * image_area:
        return image, mask

    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add small margin
    margin = 10

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    cropped = image[y1:y2, x1:x2]

    return cropped, mask


# --------------------------------------------------
# 2. Extract RGB visual features
# --------------------------------------------------
def extract_features(image):

    image = cv2.resize(image, (128, 128))

    # HSV colour features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist(
        [hsv], [0], None, [32], [0, 180]
    )

    hist_s = cv2.calcHist(
        [hsv], [1], None, [32], [0, 256]
    )

    hist_v = cv2.calcHist(
        [hsv], [2], None, [32], [0, 256]
    )

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    color_features = np.concatenate([
        hist_h,
        hist_s,
        hist_v
    ])

    # HOG texture/shape features
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Edge features
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_features = (
        cv2.resize(edges, (32, 32)).flatten()
        / 255.0
    )

    features = np.concatenate([
        color_features,
        hog_features,
        edge_features
    ])

    return features


# --------------------------------------------------
# 3. RGB Crop Health Analysis
# --------------------------------------------------
def analyze_rgb_image(image_path):

    original = cv2.imread(image_path)

    if original is None:
        return {
            "error": "Unable to read the image."
        }

    # Detect crop/leaf region
    crop_region, mask = detect_crop_region(original)

    # Extract features from detected region
    features = extract_features(crop_region)

    # AI prediction
    prediction = model.predict([features])[0]

    probabilities = model.predict_proba([features])[0]

    confidence = (
        float(np.max(probabilities)) * 100
    )

    if prediction == 0:
        health = "Healthy"
        stress = "Low"
    else:
        health = "Stressed"
        stress = "High"

    return {
        "analysis_type": "AI-Based RGB Crop Region Analysis",
        "health": health,
        "stress": stress,
        "confidence": round(confidence, 2),
        "region_detected": "Yes",
        "features_used": "HSV Color + HOG Texture + Edge Features"
    }


# --------------------------------------------------
# 4. Multispectral NDVI Analysis
# --------------------------------------------------
def analyze_multispectral(red_path, nir_path):

    red = cv2.imread(
        red_path,
        cv2.IMREAD_GRAYSCALE
    )

    nir = cv2.imread(
        nir_path,
        cv2.IMREAD_GRAYSCALE
    )

    if red is None or nir is None:
        return {
            "error": "Unable to read Red or NIR image."
        }

    # Align NIR image with Red image
    nir = cv2.resize(
        nir,
        (red.shape[1], red.shape[0])
    )

    red = red.astype(float)
    nir = nir.astype(float)

    denominator = nir + red + 1e-6

    ndvi = (
        (nir - red)
        / denominator
    )

    ndvi = np.nan_to_num(ndvi)

    mean_ndvi = float(
        np.mean(ndvi)
    )

    # Vegetation health classification
    if mean_ndvi >= 0.60:

        health = "Healthy Vegetation"
        stress = "Low"

    elif mean_ndvi >= 0.30:

        health = "Moderate Vegetation"
        stress = "Medium"

    elif mean_ndvi >= 0.10:

        health = "Weak Vegetation"
        stress = "High"

    else:

        health = "Very Low Vegetation"
        stress = "Very High"

    # NDVI-derived proxies
    moisture = np.clip(
        (mean_ndvi + 1) * 50,
        0,
        100
    )

    chlorophyll = np.clip(
        mean_ndvi * 100,
        0,
        100
    )

    return {
        "analysis_type": "Multispectral NDVI Analysis",
        "ndvi": round(mean_ndvi, 4),
        "health": health,
        "stress": stress,
        "moisture": round(float(moisture), 2),
        "chlorophyll": round(float(chlorophyll), 2)
    }