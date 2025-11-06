# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLE_CSV = "data/sample_oral_cancer.csv"
MODEL_OUT = "model/pipeline.joblib"

# ======================================================
# 🧠 STEP 1: Create a more realistic, weighted dataset
# ======================================================
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
        smoker = random.choices(["yes", "no"], weights=[0.35, 0.65])[0]
        alcohol = random.choices(["none", "light", "heavy"], weights=[0.6, 0.25, 0.15])[0]
        betel_quid_use = random.choice(["yes", "no"])
        symptom = random.choice(symptoms_examples)
        white_patches = random.choice(["yes", "no"])
        hpv = random.choice(["yes", "no"])
        genetics = random.choice(["yes", "no"])
        immune_compromised = random.choice(["yes", "no"])
        chronic_irritation = random.choice(["yes", "no"])
        poor_oral_hygiene = random.choice(["yes", "no"])
        diet = random.choice(["low", "moderate", "high"])
        oral_lesions = random.choice(["yes", "no"])
        difficulty_swallowing = random.choice(["yes", "no"])
        oral_condition = random.choice(["good", "moderate", "poor"])

        # ✅ Weighted risk scoring logic
        risk_score = 0
        if smoker == "yes": risk_score += 3
        if alcohol == "heavy": risk_score += 3
        if betel_quid_use == "yes": risk_score += 2
        if white_patches == "yes": risk_score += 2
        if hpv == "yes": risk_score += 1
        if genetics == "yes": risk_score += 1
        if immune_compromised == "yes": risk_score += 1
        if chronic_irritation == "yes": risk_score += 2
        if poor_oral_hygiene == "yes": risk_score += 1
        if oral_condition == "poor": risk_score += 3
        if oral_lesions == "yes": risk_score += 2
        if difficulty_swallowing == "yes": risk_score += 1
        risk_score += random.choice([0, 1])

        if risk_score <= 3:
            label = "low"
        elif risk_score <= 7:
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
    df["lifestyle_risk"] = df.apply(
        lambda row: sum([
            row["smoker"] == "yes",
            row["alcohol"] == "heavy",
            row["betel_quid_use"] == "yes"
        ]),
        axis=1
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Sample dataset written to: {path}")
    print(df['label'].value_counts())
    return df

# ======================================================
# ⚙️ STEP 2: Train + Evaluate the RandomForest model
# ======================================================
def train_and_save(path=SAMPLE_CSV, out=MODEL_OUT):
    df = make_sample_dataset(path)
    X = df[[
        "age", "gender", "smoker", "alcohol", "betel_quid_use",
        "white_patches", "hpv", "genetics", "immune_compromised",
        "chronic_irritation", "poor_oral_hygiene", "diet",
        "oral_lesions", "difficulty_swallowing", "oral_condition",
        "symptoms_text", "lifestyle_risk"
    ]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["age", "lifestyle_risk"]
    categorical_features = [
        "gender", "smoker", "alcohol", "betel_quid_use",
        "white_patches", "hpv", "genetics", "immune_compromised",
        "chronic_irritation", "poor_oral_hygiene", "diet",
        "oral_lesions", "difficulty_swallowing", "oral_condition"
    ]
    text_feature = "symptoms_text"

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    text_transformer = Pipeline(steps=[("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 3)))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("txt", text_transformer, text_feature)
    ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample"
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n📊 Evaluation on test set:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Confusion Matrix - Oral Cancer Risk Prediction")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved as static/confusion_matrix.png")

    # Loss vs Epochs
    epochs = np.arange(1, 21)
    loss = np.exp(-epochs / 5) + np.random.normal(0, 0.02, size=epochs.shape)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, marker='o')
    plt.title("Loss vs Epochs (Simulated)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/loss_vs_epochs.png")
    plt.close()
    print("✅ Loss vs Epochs graph saved as static/loss_vs_epochs.png")

    # Accuracy vs Epochs
    accuracy = 0.6 + 0.4 * (1 - np.exp(-epochs / 4)) + np.random.normal(0, 0.01, size=epochs.shape)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accuracy, color='green', marker='o')
    plt.title("Accuracy vs Epochs (Simulated)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/accuracy_vs_epochs.png")
    plt.close()
    print("✅ Accuracy vs Epochs graph saved as static/accuracy_vs_epochs.png")

    # =====================================
    # 📋 Epoch Metrics Table Visualization
    # =====================================
    np.random.seed(42)
    epochs = np.arange(1, 11)
    acc = 0.7 + 0.03 * np.arange(10) + np.random.normal(0, 0.005, 10)
    precision = 0.68 + 0.03 * np.arange(10) + np.random.normal(0, 0.006, 10)
    recall = 0.66 + 0.035 * np.arange(10) + np.random.normal(0, 0.005, 10)
    f1 = (2 * precision * recall) / (precision + recall)

    metrics_df = pd.DataFrame({
        "Epoch": epochs,
        "Accuracy": acc.round(3),
        "Precision": precision.round(3),
        "Recall": recall.round(3),
        "F1-Score": f1.round(3)
    })

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis('off')
    tbl = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.3)
    plt.title("Model Performance Metrics per Epoch (Simulated)", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig("static/epoch_metrics_table.png", bbox_inches='tight')
    plt.close()
    print("✅ Epoch metrics table saved as static/epoch_metrics_table.png")

    # Save model
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(clf, out)
    print(f"💾 Saved trained pipeline to: {out}")

# ======================================================
# 🚀 MAIN ENTRY
# ======================================================
if __name__ == "__main__":
    train_and_save()
