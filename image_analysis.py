import cv2
import numpy as np
import joblib
from skimage.feature import hog


# Load trained AI model
model = joblib.load("crop_health_model.pkl")


# ---------------------------------------------------------
# CROP REGION DETECTION
# ---------------------------------------------------------
def detect_crop_region(image):

    resized = cv2.resize(image, (512, 512))

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 30, 20])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest_contour
    )

    image_area = 512 * 512

    if area < 0.05 * image_area:
        return image

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    margin = 10

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        512,
        x + w + margin
    )

    y2 = min(
        512,
        y + h + margin
    )

    return resized[
        y1:y2,
        x1:x2
    ]


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_features(image):

    image = cv2.resize(
        image,
        (128, 128)
    )

    # HSV color features
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hist_h = cv2.calcHist(
        [hsv],
        [0],
        None,
        [32],
        [0, 180]
    )

    hist_s = cv2.calcHist(
        [hsv],
        [1],
        None,
        [32],
        [0, 256]
    )

    hist_v = cv2.calcHist(
        [hsv],
        [2],
        None,
        [32],
        [0, 256]
    )

    hist_h = cv2.normalize(
        hist_h,
        hist_h
    ).flatten()

    hist_s = cv2.normalize(
        hist_s,
        hist_s
    ).flatten()

    hist_v = cv2.normalize(
        hist_v,
        hist_v
    ).flatten()

    color_features = np.concatenate(
        [
            hist_h,
            hist_s,
            hist_v
        ]
    )

    # HOG texture features
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
        cv2.resize(
            edges,
            (32, 32)
        ).flatten() / 255.0
    )

    return np.concatenate(
        [
            color_features,
            hog_features,
            edge_features
        ]
    )


# ---------------------------------------------------------
# RGB IMAGE ANALYSIS
# ---------------------------------------------------------
def analyze_rgb_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return {
            "error": "Unable to read the image."
        }

    crop_region = detect_crop_region(
        image
    )

    features = extract_features(
        crop_region
    )

    prediction = model.predict(
        [features]
    )[0]

    probabilities = model.predict_proba(
        [features]
    )[0]

    confidence = (
        float(np.max(probabilities))
        * 100
    )

    if prediction == 0:

        health = "Healthy"
        stress = "Low"

        recommendation = (
            "Crop condition is good. "
            "Continue regular irrigation "
            "and monitoring."
        )

    else:

        health = "Stressed"
        stress = "High"

        recommendation = (
            "Crop needs attention. "
            "Check irrigation, nutrients, "
            "and possible disease symptoms."
        )

    return {

        "analysis_type":
            "AI-Based RGB Crop Region Analysis",

        "health":
            health,

        "stress":
            stress,

        "confidence":
            round(confidence, 2),

        "region_detected":
            "Yes",

        "features_used":
            "HSV Color + HOG Texture + Edge Features",

        "recommendation":
            recommendation
    }


# ---------------------------------------------------------
# MULTI-SPECTRAL NDVI ANALYSIS
# ---------------------------------------------------------
def analyze_multispectral(
    red_path,
    nir_path
):

    # Read Red image
    red = cv2.imread(
        red_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Read NIR image
    nir = cv2.imread(
        nir_path,
        cv2.IMREAD_GRAYSCALE
    )

    if red is None or nir is None:

        return {
            "error":
                "Unable to read Red or NIR image."
        }

    # Resize NIR to match Red image
    nir = cv2.resize(
        nir,
        (
            red.shape[1],
            red.shape[0]
        )
    )

    # Convert to floating point
    red = red.astype(float)
    nir = nir.astype(float)

    # -----------------------------------------------------
    # NDVI CALCULATION
    # -----------------------------------------------------

    denominator = (
        nir + red + 1e-6
    )

    ndvi = (
        (nir - red)
        / denominator
    )

    ndvi = np.nan_to_num(
        ndvi
    )

    mean_ndvi = float(
        np.mean(ndvi)
    )

    # -----------------------------------------------------
    # VEGETATION / HEALTH CLASSIFICATION
    # -----------------------------------------------------

    if mean_ndvi >= 0.60:

        health = "Healthy Vegetation"
        stress = "Low"

        recommendation = (
            "Vegetation condition is good. "
            "Continue regular irrigation "
            "and crop monitoring."
        )

    elif mean_ndvi >= 0.30:

        health = "Moderate Vegetation"
        stress = "Medium"

        recommendation = (
            "Monitor the crop closely. "
            "Check irrigation, nutrient "
            "availability, and plant condition."
        )

    elif mean_ndvi >= 0.10:

        health = "Weak Vegetation"
        stress = "High"

        recommendation = (
            "Crop needs attention. "
            "Check irrigation, nutrients, "
            "and possible disease symptoms."
        )

    else:

        health = "Very Low Vegetation"
        stress = "Very High"

        recommendation = (
            "Severe vegetation stress detected. "
            "Check water availability, nutrient "
            "deficiency, and possible disease."
        )

    # -----------------------------------------------------
    # MOISTURE PROXY
    # -----------------------------------------------------

    moisture = np.clip(
        (mean_ndvi + 1) * 50,
        0,
        100
    )

    # -----------------------------------------------------
    # CHLOROPHYLL PROXY
    # -----------------------------------------------------

    chlorophyll = np.clip(
        mean_ndvi * 100,
        0,
        100
    )

    # -----------------------------------------------------
    # NDVI HEATMAP
    # -----------------------------------------------------

    ndvi_visual = np.clip(
        ((ndvi + 1) / 2) * 255,
        0,
        255
    ).astype(np.uint8)

    ndvi_heatmap = cv2.applyColorMap(
        ndvi_visual,
        cv2.COLORMAP_JET
    )

    heatmap_path = (
        "uploads/ndvi_heatmap.jpg"
    )

    cv2.imwrite(
        heatmap_path,
        ndvi_heatmap
    )

    # -----------------------------------------------------
    # CHART DATA
    # -----------------------------------------------------

    chart_data = {

        "ndvi":
            round(mean_ndvi, 4),

        "moisture":
            round(
                float(moisture),
                2
            ),

        "chlorophyll":
            round(
                float(chlorophyll),
                2
            )
    }

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    return {

        "analysis_type":
            "Multispectral NDVI Analysis",

        "ndvi":
            round(mean_ndvi, 4),

        "health":
            health,

        "stress":
            stress,

        "moisture":
            round(
                float(moisture),
                2
            ),

        "chlorophyll":
            round(
                float(chlorophyll),
                2
            ),

        "heatmap":
            heatmap_path,

        "chart_data":
            chart_data,

        "recommendation":
            recommendation
    }


# ---------------------------------------------------------
# ANALYSIS CHART DATA
# ---------------------------------------------------------
def create_analysis_chart(
    ndvi,
    moisture,
    chlorophyll
):

    return {

        "NDVI":
            ndvi,

        "Moisture":
            moisture,

        "Chlorophyll":
            chlorophyll
    }