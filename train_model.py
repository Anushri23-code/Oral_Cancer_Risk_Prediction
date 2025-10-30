# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import random
import string

SAMPLE_CSV = "data/sample_oral_cancer.csv"
MODEL_OUT = "model/pipeline.joblib"

def make_sample_dataset(path, n=1000, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)
    rows = []
    symptoms_examples = [
        "white patch on inner cheek", "red patch in mouth", "mouth ulcer not healing",
        "persistent pain in mouth", "difficulty swallowing", "lump in mouth", "bleeding from mouth"
    ]
    for i in range(n):
        name = "user_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        age = random.randint(18, 80)
        gender = random.choice(["Male", "Female"])
        smoker = random.choices(["yes","no"], weights=[0.35,0.65])[0]
        alcohol = random.choices(["none","light","heavy"], weights=[0.6,0.25,0.15])[0]
        betel_quid_use = random.choice(["yes","no"])
        symptom = random.choice(symptoms_examples)
        white_patches = random.choice(["yes", "no"])
        hpv = random.choice(["yes", "no"])
        genetics = random.choice(["yes", "no"])
        immune_compromised = random.choice(["yes","no"])
        chronic_irritation = random.choice(["yes", "no"])
        poor_oral_hygiene = random.choice(["yes","no"])
        diet = random.choice(["low","moderate","high"])
        oral_lesions = random.choice(["yes","no"])
        difficulty_swallowing = random.choice(["yes","no"])
        oral_condition = random.choice(["good", "moderate", "poor"])  # optional, for risk calculation

        # Risk score calculation (simplified example)
        risk_score = 0
        for val in [white_patches, hpv, genetics, chronic_irritation, smoker, alcohol=="heavy", oral_condition=="poor"]:
            if val:
                risk_score += 1

        if risk_score <= 1:
            label = "low"
        elif risk_score <= 3:
            label = "medium"
        else:
            label = "high"

        rows.append({
            "name": name,
            "age": age,
            "gender": gender,
            "smoker": smoker,
            "alcohol": alcohol,
            "betel_quid_use": betel_quid_use,
            "white_patches": white_patches,
            "hpv": hpv,
            "genetics": genetics,
            "immune_compromised": immune_compromised,
            "chronic_irritation": chronic_irritation,
            "poor_oral_hygiene": poor_oral_hygiene,
            "diet": diet,
            "oral_lesions": oral_lesions,
            "difficulty_swallowing": difficulty_swallowing,
            "oral_condition": oral_condition,
            "symptoms_text": symptom,
            "label": label
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample dataset written to: {path}")
    return df

def train_and_save(path=SAMPLE_CSV, out=MODEL_OUT):
    df = make_sample_dataset(path)
    X = df[[
        "age", "gender", "smoker", "alcohol", "betel_quid_use",
        "white_patches", "hpv", "genetics", "immune_compromised",
        "chronic_irritation", "poor_oral_hygiene", "diet",
        "oral_lesions", "difficulty_swallowing", "oral_condition", "symptoms_text"
    ]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = ["age"]
    categorical_features = [
        "gender", "smoker", "alcohol", "betel_quid_use",
        "white_patches", "hpv", "genetics", "immune_compromised",
        "chronic_irritation", "poor_oral_hygiene", "diet",
        "oral_lesions", "difficulty_swallowing", "oral_condition"
    ]
    text_feature = "symptoms_text"

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    text_transformer = Pipeline(steps=[("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1,2)))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("txt", text_transformer, text_feature)
    ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Evaluation on test set:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(clf, out)
    print(f"Saved trained pipeline to: {out}")

if __name__ == "__main__":
    train_and_save()
