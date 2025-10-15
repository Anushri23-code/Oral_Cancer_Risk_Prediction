from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime
import csv

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/pipeline.joblib"
model = joblib.load(MODEL_PATH)

# Function to save prediction to CSV
def save_prediction(record):
    os.makedirs("data", exist_ok=True)
    csv_path = "data/predictions.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

# Main route for prediction
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect all form inputs
        name = request.form.get("name")
        age = request.form.get("age", type=int)
        gender = request.form.get("gender")
        smoker = request.form.get("smoker")
        alcohol = request.form.get("alcohol")
        white_patches = request.form.get("white_patches")
        hpv = request.form.get("hpv")
        genetics = request.form.get("genetics")
        chronic_irritation = request.form.get("chronic_irritation")
        oral_condition = request.form.get("oral_condition")
        symptoms_text = request.form.get("symptoms_text", "")

        # Prepare dataframe for model
        X_input = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "smoker": smoker,
            "alcohol": alcohol,
            "white_patches": white_patches,
            "hpv": hpv,
            "genetics": genetics,
            "chronic_irritation": chronic_irritation,
            "oral_condition": oral_condition,
            "symptoms_text": symptoms_text
        }])

        # Make prediction
        probs = model.predict_proba(X_input)[0]
        classes = model.classes_
        pred_label = classes[probs.argmax()]
        pred_prob = probs.max()

        # Save record
        record = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "age": int(age),
            "gender": gender,
            "smoker": smoker,
            "alcohol": alcohol,
            "white_patches": white_patches,
            "hpv": hpv,
            "genetics": genetics,
            "chronic_irritation": chronic_irritation,
            "oral_condition": oral_condition,
            "symptoms_text": symptoms_text,
            "predicted_label": pred_label,
            "predicted_prob": float(pred_prob)
        }
        save_prediction(record)

        # Render result page
        return render_template("result.html",
                               name=name,
                               label=pred_label,
                               prob=round(pred_prob, 2),
                               probs=dict(zip(classes, [round(p, 3) for p in probs])),
                               record=record)
    return render_template("index.html")

# Route to view prediction history
@app.route("/history")
def history():
    csv_path = "data/predictions.csv"
    if not os.path.exists(csv_path):
        return render_template("history.html", records=[])

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)

    # Show newest first
    return render_template("history.html", records=records[::-1])

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
