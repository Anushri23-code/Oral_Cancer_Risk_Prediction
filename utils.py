# utils.py
import sqlite3
import os
import pandas as pd

DB_FILE = "results/predictions.db"

def ensure_db(db_file=DB_FILE):
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        country TEXT,
        smoker TEXT,
        alcohol TEXT,
        betel_quid_use TEXT,
        hpv TEXT,
        genetics TEXT,
        immune_compromised TEXT,
        chronic_irritation TEXT,
        poor_oral_hygiene TEXT,
        diet TEXT,
        oral_lesions TEXT,
        white_patches TEXT,
        difficulty_swallowing TEXT,
        symptoms_text TEXT,
        predicted_label TEXT,
        predicted_prob REAL
    )
    """)
    conn.commit()
    conn.close()

def save_prediction(record, db_file=DB_FILE):
    ensure_db(db_file)
    conn = sqlite3.connect(db_file)
    df = pd.DataFrame([record])
    df.to_sql("predictions", conn, if_exists="append", index=False)
    conn.close()

def load_all_predictions(db_file=DB_FILE):
    ensure_db(db_file)
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df
