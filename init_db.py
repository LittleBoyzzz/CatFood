"""
Initialize SQLite Database — Full Version
===========================================
โหลดข้อมูล cleaned + clustered เข้าสู่ SQLite database
พร้อมผลลัพธ์โมเดล Supervised + Unsupervised
"""

import sqlite3
import pandas as pd
import joblib
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "catfood.db")
CSV_CLEAN = os.path.join(os.path.dirname(__file__), "data", "CatFood_cleaned.csv")
CSV_CLUSTER = os.path.join(os.path.dirname(__file__), "data", "CatFood_clustered.csv")
UNSUP_PKL = os.path.join(os.path.dirname(__file__), "output", "unsup_model.pkl")


def init_database():
    print("=" * 50)
    print("Initializing SQLite Database (Full)")
    print("=" * 50)

    conn = sqlite3.connect(DB_PATH)

    # 1. Survey responses (cleaned)
    df_clean = pd.read_csv(CSV_CLEAN)
    df_clean.to_sql("survey_responses", conn, if_exists="replace", index=True, index_label="id")
    print(f"Table survey_responses: {len(df_clean)} rows")

    # 2. Clustered data
    if os.path.exists(CSV_CLUSTER):
        df_cluster = pd.read_csv(CSV_CLUSTER)
        df_cluster.to_sql("clustered_data", conn, if_exists="replace", index=True, index_label="id")
        print(f"Table clustered_data: {len(df_cluster)} rows")

    # 3. Model results (supervised)
    conn.execute("DROP TABLE IF EXISTS model_results")
    conn.execute("""
        CREATE TABLE model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL, precision_score REAL, recall REAL,
            f1_score REAL, cv_accuracy REAL, roc_auc REAL,
            is_best INTEGER DEFAULT 0
        )
    """)
    model_data = [
        ("DecisionTree",       0.7119, 0.8710, 0.6750, 0.7606, 0.8367, 0.729, 0),
        ("RandomForest",       0.8814, 0.9024, 0.9250, 0.9136, 0.9864, 0.926, 1),
        ("LogisticRegression", 0.6441, 0.7879, 0.6500, 0.7123, 0.7822, 0.670, 0),
    ]
    conn.executemany("""
        INSERT INTO model_results
        (model_name, accuracy, precision_score, recall, f1_score, cv_accuracy, roc_auc, is_best)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, model_data)

    # 4. Cluster personas
    conn.execute("DROP TABLE IF EXISTS cluster_personas")
    conn.execute("""
        CREATE TABLE cluster_personas (
            cluster_id INTEGER PRIMARY KEY,
            name TEXT, size INTEGER, pct REAL,
            engagement TEXT, description TEXT,
            top_factor TEXT, top_pkg TEXT, top_option TEXT,
            age TEXT, gender TEXT,
            avg_factor REAL, avg_pkg REAL
        )
    """)

    if os.path.exists(UNSUP_PKL):
        unsup = joblib.load(UNSUP_PKL)
        for c, p in unsup["personas"].items():
            conn.execute("""
                INSERT INTO cluster_personas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (c, p["name"], p["size"], p["pct"], p["engagement"],
                  p["description"], p["top_factor"], p["top_pkg"],
                  p["top_option"], p["age"], p["gender"],
                  p["avg_factor"], p["avg_pkg"]))
        print(f"Table cluster_personas: {len(unsup['personas'])} rows")

    conn.commit()
    conn.close()
    print(f"\nDatabase: {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    init_database()
