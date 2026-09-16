from flask import Flask, render_template, request, send_from_directory, make_response
import os
import csv
from datetime import datetime

from image_analysis import (
    analyze_rgb_image,
    analyze_multispectral
)

app = Flask(__name__)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "analysis_history.csv"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# HISTORY COLUMNS
# =========================================================

HISTORY_FIELDS = [
    "Date",
    "Time",
    "Analysis Type",
    "Crop Health",
    "Stress Level",
    "NDVI",
    "Moisture",
    "Chlorophyll",
    "AI Confidence",
    "Recommendation"
]


# =========================================================
# CREATE HISTORY FILE
# =========================================================

def create_history_file():

    if not os.path.exists(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=HISTORY_FIELDS
            )

            writer.writeheader()

        return

    if os.path.getsize(HISTORY_FILE) == 0:

        with open(
            HISTORY_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=HISTORY_FIELDS
            )

            writer.writeheader()


# =========================================================
# SAVE ANALYSIS HISTORY
# =========================================================

def save_analysis_history(result):

    create_history_file()

    now = datetime.now()

    history_row = {
        "Date": now.strftime("%Y-%m-%d"),

        "Time": now.strftime("%H:%M:%S"),

        "Analysis Type": result.get(
            "analysis_type",
            ""
        ),

        "Crop Health": result.get(
            "health",
            ""
        ),

        "Stress Level": result.get(
            "stress",
            ""
        ),

        "NDVI": result.get(
            "ndvi",
            ""
        ),

        "Moisture": result.get(
            "moisture",
            ""
        ),

        "Chlorophyll": result.get(
            "chlorophyll",
            ""
        ),

        "AI Confidence": result.get(
            "confidence",
            ""
        ),

        "Recommendation": result.get(
            "recommendation",
            ""
        )
    }

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS
        )

        writer.writerow(history_row)

    print(
        "Analysis saved to history successfully."
    )


# =========================================================
# READ ANALYSIS HISTORY
# =========================================================

def read_analysis_history():

    create_history_file()

    history = []

    try:

        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                if not row:
                    continue

                values = [
                    str(
                        row.get(
                            field,
                            ""
                        )
                    ).strip()

                    for field in HISTORY_FIELDS
                ]

                if not any(values):
                    continue

                clean_row = {}

                for field in HISTORY_FIELDS:

                    clean_row[field] = (
                        row.get(
                            field,
                            ""
                        ) or ""
                    )

                history.append(
                    clean_row
                )

    except Exception as error:

        print(
            "History reading error:",
            error
        )

    # Newest entries first
    history.reverse()

    # -----------------------------------------------------
    # KEEP ONLY LATEST RESULT OF EACH ANALYSIS TYPE
    # -----------------------------------------------------

    latest_by_type = {}

    for item in history:

        analysis_type = item.get(
            "Analysis Type",
            ""
        )

        if analysis_type not in latest_by_type:

            latest_by_type[
                analysis_type
            ] = item

    return list(
        latest_by_type.values()
    )


# =========================================================
# SERVE UPLOADED FILES
# =========================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

@app.route(
    "/download_report"
)
def download_report():

    history = read_analysis_history()

    if not history:

        return (
            "No analysis history available yet."
        )

    latest = history[0]

    # -----------------------------------------------------
    # AI CONFIDENCE
    # -----------------------------------------------------

    confidence = latest.get(
        "AI Confidence",
        ""
    )

    if confidence:

        confidence_display = (
            confidence + "%"
        )

    else:

        confidence_display = (
            "Not Applicable"
        )


    report_date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # -----------------------------------------------------
    # HTML REPORT
    # -----------------------------------------------------

    html_report = f"""
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <title>
        Agri Vision AI Report
    </title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            color: #20352a;
            background: #ffffff;
        }}

        h1 {{
            color: #176b3a;
            text-align: center;
        }}

        h2 {{
            color: #176b3a;
            margin-top: 30px;
        }}

        .subtitle {{
            text-align: center;
            color: #60756b;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        th {{
            background: #176b3a;
            color: white;
            padding: 12px;
        }}

        td {{
            border: 1px solid #dcebe0;
            padding: 12px;
        }}

        .recommendation {{
            background: #fff8e6;
            border-left: 5px solid #e0a400;
            padding: 15px;
            margin-top: 20px;
        }}

        .accuracy {{
            font-size: 28px;
            font-weight: bold;
            color: #176b3a;
        }}

        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #60756b;
        }}

    </style>

</head>


<body>

    <h1>
        🌱 Agri Vision AI
    </h1>

    <p class="subtitle">
        Crop Health Assessment Through
        Multi-Spectral Image Analysis
    </p>

    <p class="subtitle">
        Report Generated:
        {report_date}
    </p>


    <!-- LATEST ANALYSIS -->

    <h2>
        📊 Latest Analysis
    </h2>

    <table>

        <tr>

            <th>
                Parameter
            </th>

            <th>
                Result
            </th>

        </tr>


        <tr>

            <td>
                Date
            </td>

            <td>
                {latest.get("Date", "")}
            </td>

        </tr>


        <tr>

            <td>
                Time
            </td>

            <td>
                {latest.get("Time", "")}
            </td>

        </tr>


        <tr>

            <td>
                Analysis Type
            </td>

            <td>
                {latest.get("Analysis Type", "")}
            </td>

        </tr>


        <tr>

            <td>
                Crop Health
            </td>

            <td>
                {latest.get("Crop Health", "")}
            </td>

        </tr>


        <tr>

            <td>
                Stress Level
            </td>

            <td>
                {latest.get("Stress Level", "")}
            </td>

        </tr>


        <tr>

            <td>
                NDVI
            </td>

            <td>
                {latest.get("NDVI", "")}
            </td>

        </tr>


        <tr>

            <td>
                Moisture Proxy
            </td>

            <td>
                {latest.get("Moisture", "")}%
            </td>

        </tr>


        <tr>

            <td>
                Chlorophyll Proxy
            </td>

            <td>
                {latest.get("Chlorophyll", "")}%
            </td>

        </tr>


        <tr>

            <td>
                AI Confidence
            </td>

            <td>
                {confidence_display}
            </td>

        </tr>

    </table>


    <!-- RECOMMENDATION -->

    <h2>
        💡 AI Recommendation
    </h2>

    <div class="recommendation">

        {latest.get("Recommendation", "")}

    </div>


    <!-- MODEL EVALUATION -->

    <h2>
        🤖 AI Model Evaluation
    </h2>

    <p>
        Random Forest Test Accuracy
    </p>

    <div class="accuracy">
        75.25%
    </div>

    <p>
        Precision: 76%
    </p>

    <p>
        Recall: 75%
    </p>

    <p>
        F1 Score: 75%
    </p>


    <!-- FOOTER -->

    <div class="footer">

        🌱 Agri Vision AI |
        AI-Based Crop Health Assessment

    </div>

</body>

</html>
"""


    response = make_response(
        html_report
    )

    response.headers[
        "Content-Type"
    ] = (
        "text/html; charset=utf-8"
    )

    response.headers[
        "Content-Disposition"
    ] = (
        "attachment; "
        'filename="AgriVisionAI_Report.html"'
    )

    return response


# =========================================================
# HOME PAGE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    result = None
    mode = None

    if request.method == "POST":

        mode = request.form.get(
            "mode"
        )


        # =================================================
        # RGB ANALYSIS
        # =================================================

        if mode == "rgb":

            image = request.files.get(
                "rgb_image"
            )

            if (
                image
                and image.filename != ""
            ):

                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    image.filename
                )

                image.save(
                    image_path
                )

                result = analyze_rgb_image(
                    image_path
                )

                if (
                    result
                    and not result.get(
                        "error"
                    )
                ):

                    result[
                        "uploaded_image"
                    ] = (
                        "/uploads/" +
                        image.filename
                    )

                    save_analysis_history(
                        result
                    )


        # =================================================
        # MULTISPECTRAL ANALYSIS
        # =================================================

        elif mode == "multispectral":

            red_image = request.files.get(
                "red_image"
            )

            nir_image = request.files.get(
                "nir_image"
            )

            if (
                red_image
                and nir_image
                and red_image.filename != ""
                and nir_image.filename != ""
            ):

                red_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    "red_" +
                    red_image.filename
                )

                nir_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    "nir_" +
                    nir_image.filename
                )

                red_image.save(
                    red_path
                )

                nir_image.save(
                    nir_path
                )

                result = analyze_multispectral(
                    red_path,
                    nir_path
                )

                if (
                    result
                    and not result.get(
                        "error"
                    )
                ):

                    result[
                        "red_image"
                    ] = (
                        "/uploads/red_" +
                        red_image.filename
                    )

                    result[
                        "nir_image"
                    ] = (
                        "/uploads/nir_" +
                        nir_image.filename
                    )

                    save_analysis_history(
                        result
                    )


    # =====================================================
    # LOAD HISTORY
    # =====================================================

    history = read_analysis_history()


    # =====================================================
    # DISPLAY WEBSITE
    # =====================================================

    return render_template(
        "index.html",
        result=result,
        mode=mode,
        history=history
    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    create_history_file()

    app.run(
        debug=True
    )