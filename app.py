
import re
import html
import joblib
import pandas as pd

from pathlib import Path
from flask import Flask, render_template, request
from flask import redirect, url_for
from database import (
    initialize_database,
    save_prediction,
    get_all_predictions,
    clear_predictions
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_engineer.pkl"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
feature_engineer = joblib.load(FEATURE_PATH)
initialize_database()

def clean_text(value):

    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.lower()
    value = re.sub(r"http\S+|www\S+", " ", value)
    value = re.sub(r"[^a-zA-Z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def create_job_features(form_data):

    text_fields = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    combined_text = " ".join(
        clean_text(form_data.get(field, ""))
        for field in text_fields
    )

    categorical_fields = [
        "location",
        "department",
        "salary_range",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function"
    ]

    row = {
        "combined_text": combined_text
    }

    for field in categorical_fields:
        value = form_data.get(field, "").strip()
        row[field] = value if value else None

    missing_fields = [
        "salary_range",
        "department",
        "company_profile",
        "benefits",
        "requirements",
        "industry",
        "required_education",
        "required_experience"
    ]

    for field in missing_fields:
        row[field + "_missing"] = int(
            not form_data.get(field, "").strip()
        )

    row["telecommuting"] = int(
        form_data.get("telecommuting", "0")
    )

    row["has_company_logo"] = int(
        form_data.get("has_company_logo", "0")
    )

    row["has_questions"] = int(
        form_data.get("has_questions", "0")
    )

    return pd.DataFrame([row])


@app.route("/")
def home():

    return render_template("index.html")


def generate_job_insights(form_data):

    insights = []

    checks = {
        "company_profile": "Company profile is missing.",
        "salary_range": "Salary information is not provided.",
        "requirements": "Job requirements are missing.",
        "benefits": "Employee benefits are not specified.",
        "industry": "Company industry is not specified.",
        "required_experience": "Required experience is not specified."
    }

    for field, message in checks.items():
        if not form_data.get(field, "").strip():
            insights.append(message)

    if form_data.get("has_company_logo", "0") == "0":
        insights.append("No company logo was indicated.")

    if form_data.get("description", "").strip() == "":
        insights.append("Job description is missing.")

    if not insights:
        insights.append(
            "No missing-information checks were triggered."
        )

    return insights


@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form

    job_features = create_job_features(form_data)

    transformed_features = feature_engineer.transform(
        job_features
    )

    probability = model.predict_proba(
        transformed_features
    )[0][1]

    threshold = 0.50

    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "Potentially Fraudulent"
        risk_level = "High"
    else:
        result = "Likely Genuine"
        risk_level = "Low"

    fraud_percentage = round(probability * 100, 2)
    insights = generate_job_insights(form_data)
    save_prediction(
    job_title=form_data.get("title", "Untitled Job"),
    prediction=result,
    fraud_probability=fraud_percentage
    )
    return render_template(
     "result.html",
     result=result,
     risk_level=risk_level,
     fraud_percentage=fraud_percentage,
     prediction=prediction,
     insights=insights
)



@app.route("/dashboard")
def dashboard():

    dataset_path = BASE_DIR / "dataset" / "fake_job_postings.csv"
    comparison_path = BASE_DIR / "models" / "model_comparison.csv"

    dataset = pd.read_csv(dataset_path)
    comparison = pd.read_csv(comparison_path)

    total_jobs = len(dataset)
    fraudulent_jobs = int(dataset["fraudulent"].sum())
    genuine_jobs = total_jobs - fraudulent_jobs

    fraud_percentage = round(
        (fraudulent_jobs / total_jobs) * 100, 2
    )

    genuine_percentage = round(
        (genuine_jobs / total_jobs) * 100, 2
    )

    comparison = comparison.sort_values(
        by="F1 Score",
        ascending=False
    )

    best_model = comparison.iloc[0]

    return render_template(
        "dashboard.html",
        total_jobs=total_jobs,
        fraudulent_jobs=fraudulent_jobs,
        genuine_jobs=genuine_jobs,
        fraud_percentage=fraud_percentage,
        genuine_percentage=genuine_percentage,
        comparison=comparison.to_dict(orient="records"),
        best_model=best_model
    )


@app.route("/history")
def history():

    predictions = get_all_predictions()

    return render_template(
        "history.html",
        predictions=predictions
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():

    clear_predictions()

    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(debug=True)
