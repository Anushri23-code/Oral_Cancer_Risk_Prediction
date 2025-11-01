from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os
from datetime import datetime
import csv

# -----------------------
# Initialize Flask app
# -----------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management

# -----------------------
# Users CSV for persistent login
# -----------------------
USERS_FILE = "data/users.csv"

# ✅ FIXED VERSION (prevents KeyError for 'email' or 'phone')
def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    users = []
    with open(USERS_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append({
                "email": row.get("email", "").strip(),
                "phone": row.get("phone", "").strip(),
                "password": row.get("password", "")
            })
    return users


def save_user(email, phone, password):
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(USERS_FILE)
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["email", "phone", "password"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"email": email, "phone": phone, "password": password})


# -----------------------
# Load trained model
# -----------------------
MODEL_PATH = "model/pipeline.joblib"
model = joblib.load(MODEL_PATH)

# -----------------------
# Function to save prediction
# -----------------------
def save_prediction(record):
    os.makedirs("data", exist_ok=True)
    csv_path = "data/predictions.csv"

    fieldnames = [
        "username", "timestamp", "name", "age", "gender", "country",
        "smoker", "alcohol", "betel_quid_use", "hpv", "genetics",
        "immune_compromised", "chronic_irritation", "poor_oral_hygiene",
        "diet", "oral_lesions", "white_patches", "difficulty_swallowing",
        "oral_condition", "symptoms_text", "predicted_label", "predicted_prob"
    ]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for field in fieldnames:
            record.setdefault(field, "")
        writer.writerow(record)


# -----------------------
# Welcome Page
# -----------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")


# -----------------------
# Login Page
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    users = load_users()
    if request.method == "POST":
        login_type = request.form.get("login_type")
        value = request.form.get("email_or_phone")
        password = request.form.get("password")

        for user in users:
            if (
                (login_type == "email" and user["email"] == value)
                or (login_type == "phone" and user["phone"] == value)
            ) and user["password"] == password:
                session["username"] = value
                return redirect(url_for("index"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# -----------------------
# Logout Route
# -----------------------
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


# -----------------------
# Register Page
# -----------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    users = load_users()
    if request.method == "POST":
        reg_type = request.form.get("register_type")
        value = request.form.get("email_or_phone")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if password != confirm:
            return render_template("register.html", error="Passwords do not match")

        # Check existing user
        for user in users:
            if (reg_type == "email" and user["email"] == value) or (
                reg_type == "phone" and user["phone"] == value
            ):
                return render_template("register.html", error="User already exists")

        # Save user
        email = value if reg_type == "email" else ""
        phone = value if reg_type == "phone" else ""
        save_user(email, phone, password)

        return render_template("register.html", success="Registration successful! Please login.")

    return render_template("register.html")


# -----------------------
# Prediction Page
# -----------------------
@app.route("/index", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        username = session["username"]
        name = request.form.get("name")
        age = request.form.get("age", type=int)
        gender = request.form.get("gender")
        country = request.form.get("country", "")
        smoker = request.form.get("smoker")
        alcohol = request.form.get("alcohol")
        betel_quid_use = request.form.get("betel_quid_use")
        hpv = request.form.get("hpv")
        genetics = request.form.get("genetics")
        immune_compromised = request.form.get("immune_compromised")
        chronic_irritation = request.form.get("chronic_irritation")
        poor_oral_hygiene = request.form.get("poor_oral_hygiene")
        diet = request.form.get("diet")
        oral_lesions = request.form.get("oral_lesions")
        white_patches = request.form.get("white_patches")
        difficulty_swallowing = request.form.get("difficulty_swallowing")
        oral_condition = request.form.get("oral_condition")
        symptoms_text = request.form.get("symptoms_text", "")

        X_input = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "smoker": smoker,
            "alcohol": alcohol,
            "betel_quid_use": betel_quid_use,
            "hpv": hpv,
            "genetics": genetics,
            "immune_compromised": immune_compromised,
            "chronic_irritation": chronic_irritation,
            "poor_oral_hygiene": poor_oral_hygiene,
            "diet": diet,
            "oral_lesions": oral_lesions,
            "white_patches": white_patches,
            "difficulty_swallowing": difficulty_swallowing,
            "oral_condition": oral_condition,
            "symptoms_text": symptoms_text
        }])

        probs = model.predict_proba(X_input)[0]
        classes = model.classes_
        pred_label = classes[probs.argmax()]
        pred_prob = probs.max()

        record = {
            "username": username,
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "age": age,
            "gender": gender,
            "country": country,
            "smoker": smoker,
            "alcohol": alcohol,
            "betel_quid_use": betel_quid_use,
            "hpv": hpv,
            "genetics": genetics,
            "immune_compromised": immune_compromised,
            "chronic_irritation": chronic_irritation,
            "poor_oral_hygiene": poor_oral_hygiene,
            "diet": diet,
            "oral_lesions": oral_lesions,
            "white_patches": white_patches,
            "difficulty_swallowing": difficulty_swallowing,
            "oral_condition": oral_condition,
            "symptoms_text": symptoms_text,
            "predicted_label": pred_label,
            "predicted_prob": float(pred_prob)
        }
        save_prediction(record)

        return render_template(
            "result.html",
            name=name,
            label=pred_label,
            prob=round(pred_prob, 2),
            probs=dict(zip(classes, [round(p, 3) for p in probs])),
            record=record
        )

    return render_template("index.html")


# -----------------------
# Prediction History Page
# -----------------------
@app.route("/history")
def history():
    csv_path = "data/predictions.csv"
    if not os.path.exists(csv_path):
        return render_template("history.html", records=[])

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = [row for row in reader if any(row.values())]

        if not records:
            return render_template("history.html", records=[])

        expected_fields = [
            "timestamp", "name", "age", "gender", "country",
            "smoker", "alcohol", "betel_quid_use", "hpv", "genetics",
            "immune_compromised", "chronic_irritation", "poor_oral_hygiene",
            "diet", "oral_lesions", "white_patches", "difficulty_swallowing",
            "oral_condition", "symptoms_text", "predicted_label", "predicted_prob"
        ]

        all_keys = set(expected_fields)
        for r in records:
            for k in all_keys:
                r.setdefault(k, "")

        ordered_records = [
            {key: r.get(key, "") for key in expected_fields}
            for r in records
        ]

        return render_template("history.html", records=ordered_records[::-1])
    except Exception as e:
        return f"<h3 style='color:red;'>Error loading history: {e}</h3>"


# -----------------------
# Run Flask app
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
